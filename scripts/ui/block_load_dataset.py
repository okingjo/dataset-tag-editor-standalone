from __future__ import annotations
from typing import TYPE_CHECKING, Callable
import gradio as gr

import settings
from .ui_common import *
from .uibase import UIBase

if TYPE_CHECKING:
    from .ui_classes import *


class LoadDatasetUI(UIBase):

    def create_ui_global(self, cfg_general, toprow=None):
        """Compact dataset bar rendered above the workflow tabs: every tab
        operates on the folder loaded here. Left module = load controls;
        right module = save controls (button on top, checkboxes below)."""
        with gr.Row(elem_classes=["dataset-bar"]):
            with gr.Row(elem_classes=["dataset-bar-load"]):
                self.tb_img_directory = gr.Textbox(
                    label="Dataset directory",
                    placeholder="C:\\directory\\of\\datasets",
                    value=cfg_general.dataset_dir,
                    scale=4,
                    min_width=200,
                )
                self.tb_caption_file_ext = gr.Textbox(
                    label="Caption ext",
                    placeholder=".txt (on Load and Save)",
                    value=cfg_general.caption_ext,
                    scale=1,
                    min_width=90,
                )
                self.btn_load_datasets = gr.Button(
                    value="Load", variant="primary", scale=0, min_width=70
                )
                self.btn_unload_datasets = gr.Button(
                    value="Unload", scale=0, min_width=70
                )
            if toprow is not None:
                with gr.Column(elem_classes=["dataset-bar-save"], scale=2, min_width=300):
                    toprow.create_ui_inline(cfg_general)
        if toprow is not None:
            toprow.create_ui_rest(cfg_general)

    def create_ui(self, cfg_general):
        with gr.Column(variant="panel"):
            with gr.Accordion(label="Dataset Load Settings"):
                with gr.Row():
                    with gr.Column():
                        self.cb_load_recursive = gr.Checkbox(
                            value=cfg_general.load_recursive,
                            label="Load from subdirectories",
                        )
                        self.cb_load_caption_from_filename = gr.Checkbox(
                            value=cfg_general.load_caption_from_filename,
                            label="Load caption from filename if no text file exists",
                        )
                        self.cb_replace_new_line_with_comma = gr.Checkbox(
                            value=cfg_general.replace_new_line,
                            label="Replace new-line character with comma",
                        )
                    with gr.Column():
                        self.rb_use_interrogator = gr.Radio(
                            choices=[
                                "No",
                                "If Empty",
                                "Overwrite",
                                "Prepend",
                                "Append",
                            ],
                            value=cfg_general.use_interrogator,
                            label="Use Interrogator Caption",
                        )
            with gr.Accordion(label="Interrogator", open=False):
                self.dd_intterogator_names = gr.Dropdown(
                    label="Interrogators",
                    choices=dte_instance.INTERROGATOR_NAMES,
                    value=cfg_general.use_interrogator_names,
                    interactive=True,
                    multiselect=True,
                )
                gr.Markdown(
                    "Tag the **currently filtered images** with the selected "
                    "interrogators. Merge method comes from *Use Interrogator "
                    "Caption* in Dataset Load Settings (If Empty / Overwrite / "
                    "Prepend / Append)."
                )
                with gr.Row():
                    self.btn_interrogate = gr.Button(
                        value="Start tagging", variant="primary", scale=1
                    )
                    self.txt_interrogate_status = gr.Textbox(
                        label="Status", interactive=False, scale=3
                    )
                with gr.Row():
                    self.sl_custom_threshold_z3d = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=cfg_general.custom_threshold_z3d,
                        step=0.01,
                        interactive=True,
                        label="Z3D-E621 Score Threshold",
                    )
                with gr.Row():
                    self.sl_custom_threshold_pixai = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=cfg_general.custom_threshold_pixai,
                        step=0.01,
                        interactive=True,
                        label="PixAI General Tag Threshold (other categories use official defaults)",
                    )
                with gr.Row():
                    self.cb_use_custom_threshold_waifu = gr.Checkbox(
                        value=cfg_general.use_custom_threshold_waifu,
                        label="Use Custom Threshold (WD Tagger)",
                        interactive=True,
                    )
                    self.sl_custom_threshold_waifu = gr.Slider(
                        minimum=0,
                        maximum=1,
                        value=cfg_general.custom_threshold_waifu,
                        step=0.01,
                        interactive=True,
                        label="WD Tagger Score Threshold",
                    )

    def set_callbacks(
        self,
        o_update_filter_and_gallery: list[gr.components.Component],
        toprow: ToprowUI,
        dataset_gallery: DatasetGalleryUI,
        filter_by_tags: FilterByTagsUI,
        filter_by_selection: FilterBySelectionUI,
        batch_edit_captions: BatchEditCaptionsUI,
        update_filter_and_gallery: Callable[[], list],
        get_filters: Callable[[], list],
    ):
        def load_files_from_dir(
            dir: str,
            caption_file_ext: str,
            recursive: bool,
            load_caption_from_filename: bool,
            replace_new_line: bool,
            use_interrogator: str,
            use_interrogator_names: list[str],
            use_custom_threshold_waifu: bool,
            custom_threshold_waifu: float,
            custom_threshold_z3d: float,
            custom_threshold_pixai: float,
            use_kohya_metadata: bool,
            kohya_json_path: str,
        ):
            interrogate_method = dte_instance.InterrogateMethod.NONE
            if use_interrogator == "If Empty":
                interrogate_method = dte_instance.InterrogateMethod.PREFILL
            elif use_interrogator == "Overwrite":
                interrogate_method = dte_instance.InterrogateMethod.OVERWRITE
            elif use_interrogator == "Prepend":
                interrogate_method = dte_instance.InterrogateMethod.PREPEND
            elif use_interrogator == "Append":
                interrogate_method = dte_instance.InterrogateMethod.APPEND

            threshold_waifu = custom_threshold_waifu if use_custom_threshold_waifu else -1
            threshold_z3d = custom_threshold_z3d
            threshold_pixai = custom_threshold_pixai

            dte_instance.load_dataset(
                dir,
                caption_file_ext,
                recursive,
                load_caption_from_filename,
                replace_new_line,
                interrogate_method,
                use_interrogator_names,
                threshold_waifu,
                threshold_z3d,
                threshold_pixai,
                settings.current.use_temp_files,
                kohya_json_path if use_kohya_metadata else None,
                settings.current.max_resolution
            )
            # [FIX] Allow gradio to serve images from the folder just loaded.
            # The /file= route only serves paths under blocks.allowed_paths; a
            # dataset outside the app dir / configured whitelist otherwise shows
            # up as blank boxes (HTTP 403). Append the loaded dir dynamically.
            try:
                import interface as _iface
                from pathlib import Path as _Path
                _blocks = _iface.interface
                if _blocks is not None:
                    _dir = str(_Path(dir).absolute())
                    if _dir not in _blocks.allowed_paths:
                        _blocks.allowed_paths.append(_dir)
            except Exception:
                pass
            # [PERF] paginate the initial gallery view
            dataset_gallery.reset_page()
            imgs = dte_instance.get_filtered_imgs(filters=[])
            page_imgs, _, _ = dataset_gallery.apply_pagination(imgs)
            return (
                [page_imgs, []]
                + update_filter_and_gallery()
            )

        self.btn_load_datasets.click(
            fn=load_files_from_dir,
            inputs=[
                self.tb_img_directory,
                self.tb_caption_file_ext,
                self.cb_load_recursive,
                self.cb_load_caption_from_filename,
                self.cb_replace_new_line_with_comma,
                self.rb_use_interrogator,
                self.dd_intterogator_names,
                self.cb_use_custom_threshold_waifu,
                self.sl_custom_threshold_waifu,
                self.sl_custom_threshold_z3d,
                self.sl_custom_threshold_pixai,
                toprow.cb_save_kohya_metadata,
                toprow.tb_metadata_output,
            ],
            outputs=[
                dataset_gallery.gl_dataset_images,
                filter_by_selection.gl_filter_images,
            ]
            + o_update_filter_and_gallery,
        )

        def unload_files():
            dte_instance.clear()
            return (
                [[], []]
                + filter_by_tags.clear_filters()
                + [batch_edit_captions.tag_select_ui_remove.cbg_tags_update()]
            )

        self.btn_unload_datasets.click(
            fn=unload_files,
            outputs=[
                dataset_gallery.gl_dataset_images,
                filter_by_selection.gl_filter_images,
            ]
            + filter_by_tags.clear_filters_output()
            + [batch_edit_captions.tag_select_ui_remove.cbg_tags]
        ).then(
            fn=lambda:update_filter_and_gallery(),
            outputs=o_update_filter_and_gallery
        )

        def interrogate_filtered(
            use_interrogator: str,
            use_interrogator_names: list[str],
            use_custom_threshold_waifu: bool,
            custom_threshold_waifu: float,
            custom_threshold_z3d: float,
            custom_threshold_pixai: float,
            progress=gr.Progress(),
        ):
            if len(dte_instance.dataset) == 0:
                return "Load a dataset first."
            interrogate_method = dte_instance.InterrogateMethod.NONE
            if use_interrogator == "If Empty":
                interrogate_method = dte_instance.InterrogateMethod.PREFILL
            elif use_interrogator == "Overwrite":
                interrogate_method = dte_instance.InterrogateMethod.OVERWRITE
            elif use_interrogator == "Prepend":
                interrogate_method = dte_instance.InterrogateMethod.PREPEND
            elif use_interrogator == "Append":
                interrogate_method = dte_instance.InterrogateMethod.APPEND
            if interrogate_method == dte_instance.InterrogateMethod.NONE:
                return 'Set "Use Interrogator Caption" (Dataset Load Settings) to a method other than No.'
            if not use_interrogator_names:
                return "Select at least one interrogator."
            paths = dte_instance.get_filtered_imgpaths(filters=get_filters())
            if not paths:
                return "No images match the current filters."
            threshold_waifu = custom_threshold_waifu if use_custom_threshold_waifu else -1

            def progress_cb(done, total, name):
                progress(done / total, desc=f"{name}: {done}/{total}")

            processed, skipped = dte_instance.interrogate_loaded(
                paths,
                interrogate_method,
                use_interrogator_names,
                threshold_waifu,
                custom_threshold_z3d,
                custom_threshold_pixai,
                progress_cb=progress_cb,
            )
            return f"Done: {processed} images tagged, {skipped} skipped. Save all changes to write the captions."

        self.btn_interrogate.click(
            fn=interrogate_filtered,
            inputs=[
                self.rb_use_interrogator,
                self.dd_intterogator_names,
                self.cb_use_custom_threshold_waifu,
                self.sl_custom_threshold_waifu,
                self.sl_custom_threshold_z3d,
                self.sl_custom_threshold_pixai,
            ],
            outputs=[self.txt_interrogate_status],
        ).then(
            fn=lambda: update_filter_and_gallery(),
            outputs=o_update_filter_and_gallery,
        )
