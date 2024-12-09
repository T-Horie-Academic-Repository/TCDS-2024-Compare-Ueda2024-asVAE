from pytorch_lightning.callbacks import Callback
from pytorch_lightning import Trainer
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Literal, Optional, Any
from collections import defaultdict
import json
import ndjson

from tqdm import tqdm

from ..data import Batch
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

    @classmethod
    def make_common_save_file_path(
        cls,
        save_dir: Path,
        dataloader_idx: int,
        is_test: bool = False,
    ):
        if is_test:
            return save_dir / f"language_dataloader_idx_{dataloader_idx}_test.jsonl"
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
        is_test: bool = False,
    ) -> None:
        game_training_state = game.training
        game.eval()

        for dataloader_idx, dataloader in enumerate(dataloaders):
            if not self.meaning_saved_flag:
                self.meaning_saved_flag = True

                meanings: list[Any] = []
                for batch in dataloader:
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
                        self.save_dir, dataloader_idx, is_test=is_test
                    ).open("w") as f:
                        ndjson_writer = ndjson.writer(f)
                        ndjson_writer.writerow(
                            {self.make_common_json_key_name("meaning", step=step): meanings}
                        )

            messages: defaultdict[tuple[int, int], list[list[int]]] = defaultdict(list)
            message_lengths: defaultdict[tuple[int, int], list[int]] = defaultdict(list)

            for batch in dataloader:
                batch: Batch = batch.to(game.device)
                for sender_idx, sender in list(enumerate(game.senders)):
                    for beam_size in self.beam_sizes:
                        sender_output = sender.forward(batch, beam_size=beam_size)
                        messages[sender_idx, beam_size].extend(
                            (sender_output.message * sender_output.message_mask.long()).tolist()
                        )
                        message_lengths[sender_idx, beam_size].extend(sender_output.message_length.tolist())

            for sender_idx, beam_size in messages.keys():
                with self.make_common_save_file_path(
                    self.save_dir, dataloader_idx, is_test=is_test
                ).open("a") as f:
                    ndjson_writer = ndjson.writer(f)
                    ndjson_writer.writerow(
                        {
                            self.make_common_json_key_name(
                                "message",
                                step=step,
                                sender_idx=sender_idx,
                                beam_size=beam_size,
                            ): messages[sender_idx, beam_size],
                            self.make_common_json_key_name(
                                "message_length",
                                step=step,
                                sender_idx=sender_idx,
                                beam_size=beam_size,
                            ): message_lengths[sender_idx, beam_size],
                        }
                    )

        game.train(game_training_state)

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

    def on_train_epoch_end(self, trainer, pl_module):
        self.pbar.update(1)

    ## added for TCDS-2024
    def on_test_end(self, trainer, pl_module):
        """
        Dumps the language data at the end of the test.
        The format of the dumped data is the same as the one in the fit phase.
        """
        self.meaning_saved_flag = False
        dataloaders: Optional[list[DataLoader[Batch]]] = trainer.test_dataloaders
        if dataloaders is None:
            return

        self.dump(game=pl_module, dataloaders=dataloaders, step="test", is_test=True)

