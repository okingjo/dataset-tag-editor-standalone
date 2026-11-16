import gradio as gr

import settings
from dte_instance import dte_instance
from dataset_tag_editor import multi_caption as mc
from dataset_tag_editor import vlm_captioner as vlm

CANDIDATE_CHOICES = ["tags only", "tags + NL (hybrid)", "NL only"]
TAG_SOURCE_CHOICES = ["Saved .txt files", "Current editor tags"]
TAG_STYLE_CHOICES = ["spaces (Anima/MOSS)", "underscores (booru)"]


class NlMultiCaptionsUI:
    """Split across two top-level tabs: NL generation lives in *Captioning*,
    .multi export lives in *Export Multi*. One instance, two create_ui_* calls."""

    def create_ui_nl(self, cfg_nl_multi):
        gr.Markdown(
            "**Natural-language captions** — generated with an OpenAI-compatible "
            "vision server (Ollama / llama.cpp / vLLM), saved per image as "
            "`<img>.nl.txt`. Review/edit them in *Tag Editor → Edit Caption of "
            "Selected Image → VLM NL Caption*, then combine with tags in the "
            "*Export Multi* tab.<br>"
            "**Per-folder context:** put a `nl_prompt.txt` file in any dataset "
            "folder to give the VLM extra instructions for all images inside it, "
            "e.g. `All images in this folder show <character> wearing <costume>.` "
            "With recursive loading, files are inherited from the dataset root "
            "down to each image's folder (character folder → costume subfolder)."
        )
        with gr.Accordion("Generate NL captions (VLM)", open=True):
            with gr.Row():
                self.tb_base_url = gr.Textbox(
                    label="Base URL (OpenAI-compatible)",
                    value=settings.current.caption_llm_base_url,
                    scale=2,
                )
                self.tb_model = gr.Textbox(
                    label="Model",
                    value=settings.current.caption_llm_model,
                    scale=2,
                )
                self.tb_api_key = gr.Textbox(
                    label="API key",
                    value=settings.current.caption_llm_api_key,
                    type="password",
                    scale=1,
                )
            with gr.Row():
                self.cb_thinking = gr.Checkbox(
                    label="Enable thinking (hybrid models)",
                    value=settings.current.caption_llm_enable_thinking,
                )
                self.cb_skip_existing_nl = gr.Checkbox(
                    label="Skip images with existing .nl.txt",
                    value=cfg_nl_multi.skip_existing_nl,
                )
                self.cb_tags_hint = gr.Checkbox(
                    label="Send existing tags as hints",
                    value=cfg_nl_multi.tags_hint,
                )
            self.tb_prompt = gr.Textbox(
                label="Custom prompt (empty = built-in anime captioning prompt)",
                value=settings.current.caption_llm_prompt,
                lines=2,
            )
            with gr.Row():
                self.btn_save_vlm_settings = gr.Button(
                    "Save VLM settings", variant="secondary"
                )
                self.btn_generate_nl = gr.Button(
                    "Generate NL captions (filtered images)", variant="primary"
                )
            self.tb_nl_log = gr.Textbox(label="Log", lines=8, interactive=False)

    def create_ui_multi(self, cfg_nl_multi):
        gr.Markdown(
            "**Export `.multi` candidates** — combines the edited `.txt` tags "
            "with the `.nl.txt` sidecars into complete caption candidates "
            "separated by `---===---` (kohya sd-scripts samples one per training "
            "step; use `caption_extension = \".multi\"` for Anima/MOSS). "
            "Re-run any time after editing — `.multi` is always re-derivable "
            "from `.txt` + `.nl.txt`. SD/Illustrious/Pony keep using the plain "
            "`.txt` files."
        )
        with gr.Accordion("Export .multi candidates", open=True):
            self.rb_tag_source = gr.Radio(
                choices=TAG_SOURCE_CHOICES,
                value=cfg_nl_multi.tag_source,
                label="Tag source",
            )
            self.cbg_candidates = gr.CheckboxGroup(
                choices=CANDIDATE_CHOICES,
                value=list(cfg_nl_multi.candidates),
                label="Candidates to include (each becomes one .multi block)",
            )
            with gr.Row():
                self.rb_tag_style = gr.Radio(
                    choices=TAG_STYLE_CHOICES,
                    value=cfg_nl_multi.tag_style,
                    label="Tag style inside .multi",
                )
                self.tb_multi_ext = gr.Textbox(
                    label="Output extension", value=cfg_nl_multi.multi_ext
                )
            self.tb_quality_prefix = gr.Textbox(
                label="Quality prefix (prepended to tag candidates)",
                value=cfg_nl_multi.quality_prefix,
            )
            with gr.Row():
                self.cb_prefix_nl = gr.Checkbox(
                    label="Prepend quality prefix to NL-only candidate",
                    value=cfg_nl_multi.prefix_nl,
                )
                self.cb_overwrite_multi = gr.Checkbox(
                    label="Overwrite existing .multi files",
                    value=cfg_nl_multi.overwrite_multi,
                )
            self.btn_generate_multi = gr.Button(
                "Generate .multi (filtered images)", variant="primary"
            )
            self.tb_multi_log = gr.Textbox(label="Log", lines=8, interactive=False)

    def set_callbacks(self, get_filters):
        import ui

        def save_vlm_settings(base_url, model, api_key, thinking, prompt):
            settings.current = settings.current._replace(
                caption_llm_base_url=base_url,
                caption_llm_model=model,
                caption_llm_api_key=api_key,
                caption_llm_enable_thinking=thinking,
                caption_llm_prompt=prompt,
            )
            settings.save()
            return "VLM settings saved to settings.json"

        self.btn_save_vlm_settings.click(
            save_vlm_settings,
            inputs=[
                self.tb_base_url,
                self.tb_model,
                self.tb_api_key,
                self.cb_thinking,
                self.tb_prompt,
            ],
            outputs=[self.tb_nl_log],
        )

        def generate_nl(
            base_url,
            model,
            api_key,
            thinking,
            skip_existing,
            tags_hint,
            prompt,
            dataset_dir,
            *filter_args,
            progress=gr.Progress(),
        ):
            filters = get_filters(*filter_args)
            imgpaths = dte_instance.get_filtered_imgpaths(filters)
            if not imgpaths:
                return "No images loaded (or nothing matches the filters)."
            opts = dict(
                base_url=base_url,
                model=model,
                api_key=api_key,
                enable_thinking=thinking,
                prompt=prompt,
                system_prompt=settings.current.caption_llm_system_prompt,
                timeout=settings.current.caption_llm_timeout,
                max_tokens=settings.current.caption_llm_max_tokens,
                image_max_side=settings.current.caption_llm_image_size,
            )
            hint_getter = (
                (lambda p: dte_instance.get_tags_by_image_path(p)) if tags_hint else None
            )
            counts = {"ok": 0, "skipped": 0, "error": 0}
            lines: list[str] = []

            def progress_cb(done, total, path):
                progress(done / total, desc=f"VLM captioning {done}/{total}")

            for path, status, text in vlm.caption_images(
                imgpaths,
                skip_existing=skip_existing,
                tags_hint_getter=hint_getter,
                progress_cb=progress_cb,
                root_dir=dataset_dir or None,
                **opts,
            ):
                counts[status] += 1
                name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                if status == "ok":
                    lines.append(f"[ok] {name}: {text[:100]}...")
                elif status == "error":
                    lines.append(f"[error] {name}: {text}")
            summary = (
                f"Done: {counts['ok']} generated, {counts['skipped']} skipped, "
                f"{counts['error']} errors (total {len(imgpaths)})."
            )
            return summary + "\n" + "\n".join(lines[-200:])

        self.btn_generate_nl.click(
            generate_nl,
            inputs=[
                self.tb_base_url,
                self.tb_model,
                self.tb_api_key,
                self.cb_thinking,
                self.cb_skip_existing_nl,
                self.cb_tags_hint,
                self.tb_prompt,
                ui.load_dataset.tb_img_directory,
            ],
            outputs=[self.tb_nl_log],
        )

        def generate_multi(
            tag_source,
            candidates_sel,
            tag_style,
            quality_prefix,
            prefix_nl,
            overwrite,
            multi_ext,
            caption_ext,
            *filter_args,
        ):
            filters = get_filters(*filter_args)
            imgpaths = dte_instance.get_filtered_imgpaths(filters)
            if not imgpaths:
                return "No images loaded (or nothing matches the filters)."
            if not candidates_sel:
                return "Select at least one candidate type."
            style = "spaces" if tag_style.startswith("spaces") else "underscores"
            counts = {"written": 0, "exists": 0, "empty": 0}
            no_nl = 0
            lines: list[str] = []
            for path in imgpaths:
                if tag_source == "Saved .txt files":
                    tags = mc.read_tags_from_txt(path, caption_ext or ".txt")
                else:
                    tags = dte_instance.get_tags_by_image_path(path)
                nl = mc.read_nl(path)
                if not nl:
                    no_nl += 1
                candidates = mc.build_candidates(
                    tags,
                    nl,
                    include_tags="tags only" in candidates_sel,
                    include_hybrid="tags + NL (hybrid)" in candidates_sel,
                    include_nl="NL only" in candidates_sel,
                    tag_style=style,
                    quality_prefix=quality_prefix,
                    prefix_nl=prefix_nl,
                )
                status = mc.write_multi(path, candidates, multi_ext or ".multi", overwrite)
                counts[status] += 1
                if status == "empty":
                    name = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
                    lines.append(f"[empty] {name}: no tags and no NL caption")
            summary = (
                f"Done: {counts['written']} .multi written, {counts['exists']} kept "
                f"(overwrite off), {counts['empty']} empty (total {len(imgpaths)})."
            )
            if no_nl:
                summary += f" {no_nl} images have no .nl.txt yet (NL candidates omitted)."
            return summary + ("\n" + "\n".join(lines[-100:]) if lines else "")

        self.btn_generate_multi.click(
            generate_multi,
            inputs=[
                self.rb_tag_source,
                self.cbg_candidates,
                self.rb_tag_style,
                self.tb_quality_prefix,
                self.cb_prefix_nl,
                self.cb_overwrite_multi,
                self.tb_multi_ext,
                ui.load_dataset.tb_caption_file_ext,
            ],
            outputs=[self.tb_multi_log],
        )
