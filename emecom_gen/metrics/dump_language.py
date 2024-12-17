from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Literal, Optional, Any
from collections import defaultdict
import ndjson

import numpy as np
from tqdm import tqdm

from ..data import Batch
from ..data.tcds_data import tidyup_receiver_output
from ..model.game import GameBase


class DumpLanguage(Callback):
    def __init__(
        self,
        save_dir: Path,
        meaning_type: Literal["input", "target_label", "path"],
        beam_sizes: tuple[int, ...] = (1, 2, 4, 8),
    ):
        super().__init__()
        self.save_dir = save_dir
        self.meaning_type: Literal["input", "target_label", "path"] = meaning_type
        self.beam_sizes = beam_sizes

        self.meaning_saved_flag = False
        self.pbar = tqdm(desc="training", leave=False, total=20000)
        self.pbar_aux = tqdm(desc="log-data", leave=False)

    @classmethod
    def make_common_save_file_path(
        cls,
        save_dir: Path,
        dataloader_idx: int,
        file_prefix: str|None = None,
    ):
        if file_prefix is not None:
            return save_dir / f"language_dataloader_idx_{dataloader_idx}_{file_prefix}.jsonl"
        return save_dir / f"language_dataloader_idx_{dataloader_idx}.jsonl"

    @classmethod
    def make_common_json_key_name(
        cls,
        key_type: Literal["meaning", "message", "message_length"],
        step: int | Literal["last"] = "last",
        sender_idx: int = 0,
        beam_size: int = 1,
    ):
        match key_type:
            case "meaning":
                return "meaning"
            case other if other in ("message", "message_length"):
                return f"{other}_step_{step}_sender_idx_{sender_idx}_beam_size_{beam_size}"
            case _:
                raise ValueError(f"Unknown key_type {key_type}.")

    def dump(
        self,
        game: GameBase,
        dataloaders: list[DataLoader[Batch]],
        step: int | Literal["last"] = "last",
        file_prefix: str | None = None,
        n_attributes: int = 4,
        n_values: int = 4,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """ output the language data to the jsonl file.
        
        Args:
            file_prefix (str | None): the prefix of the file name. If `None`, the data won't be saved.
        
        """      
        game_training_state = game.training
        game.eval()


        for dataloader_idx, dataloader in enumerate(dataloaders):
            if self.meaning_saved_flag and file_prefix is not None:
                continue
            elif file_prefix is not None:
                self.meaning_saved_flag = True

            meanings: list[Any] = []
            
            ## extraction of input data (as meaning)
            for batch in dataloader:
                if file_prefix is None:
                    meanings.extend(batch.target_label)
                    continue
                                    
                batch: Batch
                match self.meaning_type:
                    case "input":
                        meanings.extend(batch.input.tolist())
                    case "target_label":
                        meanings.extend(batch.target_label.tolist())
                    case "path":
                        assert (
                            batch.input_data_path is not None
                        ), "`batch.input_data_path` should not be `None` when `self.meaning_type == 'patch'`."
                        if isinstance(batch.input_data_path, Path):
                            meanings.append(batch.input_data_path)
                        else:
                            meanings.extend(batch.input_data_path)
                with self.make_common_save_file_path(
                    self.save_dir, dataloader_idx, file_prefix=file_prefix
                ).open("w") as f:
                    ndjson_writer = ndjson.writer(f)
                    ndjson_writer.writerow(
                        {"meaning": meanings}
                    )
                    # ndjson_writer.writerow(
                    #     {self.make_common_json_key_name("estimated_meaning", step=step): meanings}
                    # )

            messages: defaultdict[tuple[int, int], list[list[int]]] = defaultdict(list)
            message_lengths: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
            message_masks: defaultdict[tuple[int, int], list[list[int]]] = defaultdict(list)
            estimated_meanings: defaultdict[tuple[int, int], list[int]] = defaultdict(list)

            dumped_result: list[dict[str, Any]] = [] ## added for TCDS-2024
            for batch in dataloader:
                batch: Batch = batch.to(game.device)
                for agentspair_idx, (sender, receiver) in enumerate(zip(game.senders, game.receivers)):
                    for beam_size in self.beam_sizes:
                        sender_output = sender.forward(batch, beam_size=beam_size)
                        messages[agentspair_idx, beam_size].extend(
                            (sender_output.message * sender_output.message_mask.long()).tolist()
                        )
                        message_lengths[agentspair_idx, beam_size].extend(sender_output.message_length.tolist())
                        message_masks[agentspair_idx, beam_size].extend(sender_output.message_mask.tolist())
                
                        estimated_meanings[agentspair_idx, beam_size].extend(
                            tidyup_receiver_output(
                                n_attributes,
                                n_values,
                                receiver.forward(
                                    message=sender_output.message,
                                    message_length=sender_output.message_length,
                                    message_mask=sender_output.message_mask,
                                    candidates=batch.candidates,
                                ).last_logits.tolist()
                            ).tolist()
                        )

            for agentspair_idx, beam_size in messages.keys():
                dumped_result.append(
                    {
                        self.make_common_json_key_name(
                            "message",
                            step=step,
                            sender_idx=agentspair_idx,
                            beam_size=beam_size,
                        ): messages[agentspair_idx, beam_size],
                        self.make_common_json_key_name(
                            "message_length",
                            step=step,
                            sender_idx=agentspair_idx,
                            beam_size=beam_size,
                        ): message_lengths[agentspair_idx, beam_size],
                        "estimated_meaning": estimated_meanings[agentspair_idx, beam_size],
                    }
                )
                
                ## save the data
                ## If the file_prefix is None, the data won't be saved.
                if file_prefix is None:
                    continue
                
                with self.make_common_save_file_path(
                    self.save_dir, dataloader_idx, file_prefix=file_prefix
                ).open("a") as f:
                    ndjson_writer = ndjson.writer(f)
                    ndjson_writer.writerow(dumped_result[-1])

        game.train(game_training_state)
        return {"meaning": meanings}, dumped_result

    def calculate_accuracy(self, array1: np.ndarray, array2: np.ndarray) -> tuple[float, float]:
        """
        Calculate the accuracy of matching rows and individual elements between two arrays.

        Parameters:
            a (np.ndarray): First array of shape (X, Y)
            b (np.ndarray): Second array of shape (X, Y)

        Returns:
            tuple: (acc, acc_or)
                acc: The ratio of rows where all elements match between a and b.
                acc_or: The ratio of elements that match between a and b.
        """
        if array1.shape != array2.shape:
            raise ValueError("Arrays must have the same shape.")

        row_matches = np.all(array1 == array2, axis=1)
        acc = np.sum(row_matches) / array1.shape[0]

        element_matches = array1 == array2
        acc_or = np.sum(element_matches) / array1.size

        return acc, acc_or

    # Pytest test cases
    def test_calculate_accuracy(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])
        
        acc, acc_or = self.calculate_accuracy(a, b)
        assert acc == 2 / 3, f"Expected 2/3 but got {acc}"
        assert acc_or == 8 / 9, f"Expected 8/9 but got {acc_or}"

        a = np.array([[1, 1, 1], [1, 1, 1]])
        b = np.array([[1, 1, 1], [0, 0, 0]])
        acc, acc_or = self.calculate_accuracy(a, b)
        assert acc == 1 / 2, f"Expected 1/2 but got {acc}"
        assert acc_or == 3 / 6, f"Expected 3/6 but got {acc_or}"

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        acc, acc_or = self.calculate_accuracy(a, b)
        assert acc == 0.0, f"Expected 0.0 but got {acc}"
        assert acc_or == 0.0, f"Expected 0.0 but got {acc_or}"

        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])
        acc, acc_or = self.calculate_accuracy(a, b)
        assert acc == 1.0, f"Expected 1.0 but got {acc}"
        assert acc_or == 1.0, f"Expected 1.0 but got {acc_or}"
        print("finished!!")

    def on_fit_end(
        self,
        trainer: Trainer,
        pl_module: GameBase,
    ) -> None:
        dataloaders: Optional[list[DataLoader[Batch]]] = trainer.val_dataloaders

        if dataloaders is None:
            return

        self.dump(game=pl_module, dataloaders=dataloaders, step="last")
        self.pbar.close()




    ## added for TCDS-2024 ############################################################################################################
    def on_train_epoch_end(self, trainer, pl_module: GameBase):
        """ Called when the train epoch ends.
        
        This method is used for showing progress.
        
        """
        # game_loss = pl_module.loss
        self.pbar.update(1)
        self.pbar_aux.update(1)
        
        # meanings, outputs = self.dump(
        #     game=pl_module,
        #     dataloaders=[trainer.train_dataloader],
        #     file_prefix=None,
        #     n_attributes = trainer.datamodule.n_attributes,
        #     n_values = trainer.datamodule.n_values,
        # )

        # acc, acc_or = self.calculate_accuracy(
        #     np.array(outputs[0]["estimated_meaning"]),
        #     np.array(meanings["meaning"]),
        # )
        
        progeress_str = f"loss: {pl_module.loss:.6f}" #, acc: {acc:.6f}, acc_or: {acc_or:.6f}"
        self.pbar_aux.set_postfix_str(progeress_str)

        if self.pbar.n % 10 == 0:
            tqdm.write(progeress_str)
 

    # def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
    #     """ Called when the train batch ends.
        
    #     This method is used for showing the loss value of the epoch.
        
    #     """
    #     if "loss" in outputs:
    #         print(outputs["loss"].item())  # loss をリストに保存
    #     else:
    #         print("loss is not in outputs")

    def on_test_end(self, trainer, pl_module):
        """
        Dumps the language data at the end of the test.
        The format of the dumped data is the same as the one in the fit phase.
        """
        self.meaning_saved_flag = False
        dataloaders: Optional[list[DataLoader[Batch]]] = trainer.test_dataloaders
        if dataloaders is None:
            return

        file_prefix = (
            f"test-{trainer.datamodule.num_char_sorts:02}_"
            f"exp{trainer.datamodule.exp_id:03}_predict{trainer.datamodule.pred_id:03}"
        )

        self.dump(
            game=pl_module,
            dataloaders=dataloaders,
            step="test",
            file_prefix=file_prefix,
            n_attributes = trainer.datamodule.n_attributes,
            n_values = trainer.datamodule.n_values,
        )

