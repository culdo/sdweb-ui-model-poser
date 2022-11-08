import gradio as gr

import modules.codeformer_model
import modules.generation_parameters_copypaste as parameters_copypaste
import modules.gfpgan_model
import modules.hypernetworks.ui
import modules.hypernetworks.ui
import modules.ldsr_model
import modules.scripts
import modules.shared as shared
import modules.styles
import modules.textual_inversion.ui
import modules.textual_inversion.ui
from modules import script_callbacks
from modules.images import save_image
from modules.sd_samplers import samplers_for_img2img
from modules.shared import opts, cmd_opts
from modules.ui import setup_progressbar, create_output_panel, connect_reuse_seed, create_seed_inputs, gr_show, \
    interrogate, interrogate_deepbooru, roll_artist, create_toprow, add_style, apply_styles, update_token_counter
from webui import wrap_gradio_gpu_call


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as poser:
        dummy_component = gr.Label(visible=False)
        img2img_prompt, roll, img2img_prompt_style, img2img_negative_prompt, img2img_prompt_style2, submit, img2img_interrogate, img2img_deepbooru, img2img_prompt_style_apply, img2img_save_style, img2img_paste, token_counter, token_button = create_toprow(
            is_img2img=True)

        with gr.Row(elem_id='img2img_progress_row'):
            img2img_prompt_img = gr.File(label="", elem_id="img2img_prompt_image", file_count="single", type="bytes",
                                         visible=False)

            with gr.Column(scale=1):
                pass

            with gr.Column(scale=1):
                progressbar = gr.HTML(elem_id="img2img_progressbar")
                img2img_preview = gr.Image(elem_id='img2img_preview', visible=False)
                setup_progressbar(progressbar, img2img_preview, 'img2img')

        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                with gr.Tabs(elem_id="mode_img2img") as tabs_img2img_mode:
                    with gr.TabItem('img2img', id='img2img'):
                        init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False,
                                            source="upload", interactive=True, type="pil",
                                            tool=cmd_opts.gradio_img2img_tool).style(height=480)

                    with gr.TabItem('Inpaint', id='inpaint'):
                        init_img_with_mask = gr.Image(label="Image for inpainting with mask", show_label=False,
                                                      elem_id="img2maskimg", source="upload", interactive=True,
                                                      type="pil", tool="sketch", image_mode="RGBA").style(height=480)

                        init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload",
                                                    interactive=True, type="pil", visible=False,
                                                    elem_id="img_inpaint_base")
                        init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil",
                                                     visible=False, elem_id="img_inpaint_mask")

                        mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4)

                        with gr.Row():
                            mask_mode = gr.Radio(label="Mask mode", show_label=False,
                                                 choices=["Draw mask", "Upload mask"], type="index", value="Draw mask",
                                                 elem_id="mask_mode")
                            inpainting_mask_invert = gr.Radio(label='Masking mode', show_label=False,
                                                              choices=['Inpaint masked', 'Inpaint not masked'],
                                                              value='Inpaint masked', type="index")

                        inpainting_fill = gr.Radio(label='Masked content',
                                                   choices=['fill', 'original', 'latent noise', 'latent nothing'],
                                                   value='original', type="index")

                        with gr.Row():
                            inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=False)
                            inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels',
                                                                 minimum=0, maximum=256, step=4, value=32)

                    with gr.TabItem('Batch img2img', id='batch'):
                        hidden = '<br>Disabled when launched with --hide-ui-dir-config.' if shared.cmd_opts.hide_ui_dir_config else ''
                        gr.HTML(
                            f"<p class=\"text-gray-500\">Process images in a directory on the same machine where the server is running.<br>Use an empty output directory to save pictures normally instead of writing to the output directory.{hidden}</p>")
                        img2img_batch_input_dir = gr.Textbox(label="Input directory", **shared.hide_dirs)
                        img2img_batch_output_dir = gr.Textbox(label="Output directory", **shared.hide_dirs)

                with gr.Row():
                    resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", show_label=False,
                                           choices=["Just resize", "Crop and resize", "Resize and fill"], type="index",
                                           value="Just resize")

                steps = gr.Slider(minimum=1, maximum=150, step=1, label="Sampling Steps", value=20)
                sampler_index = gr.Radio(label='Sampling method', choices=[x.name for x in samplers_for_img2img],
                                         value=samplers_for_img2img[0].name, type="index")

                with gr.Group():
                    width = gr.Slider(minimum=64, maximum=2048, step=64, label="Width", value=512,
                                      elem_id="img2img_width")
                    height = gr.Slider(minimum=64, maximum=2048, step=64, label="Height", value=512,
                                       elem_id="img2img_height")

                with gr.Row():
                    restore_faces = gr.Checkbox(label='Restore faces', value=False,
                                                visible=len(shared.face_restorers) > 1)
                    tiling = gr.Checkbox(label='Tiling', value=False)

                with gr.Row():
                    batch_count = gr.Slider(minimum=1, step=1, label='Batch count', value=1)
                    batch_size = gr.Slider(minimum=1, maximum=8, step=1, label='Batch size', value=1)

                with gr.Group():
                    cfg_scale = gr.Slider(minimum=1.0, maximum=30.0, step=0.5, label='CFG Scale', value=7.0)
                    denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength',
                                                   value=0.75)

                seed, reuse_seed, subseed, reuse_subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox = create_seed_inputs()

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_img2img.setup_ui(is_img2img=True)

            img2img_gallery, generation_info, html_info = create_output_panel("img2img", opts.outdir_img2img_samples)
            parameters_copypaste.bind_buttons({"img2img": img2img_paste}, None, img2img_prompt)

            connect_reuse_seed(seed, reuse_seed, generation_info, dummy_component, is_subseed=False)
            connect_reuse_seed(subseed, reuse_subseed, generation_info, dummy_component, is_subseed=True)

            img2img_prompt_img.change(
                fn=modules.images.image_data,
                inputs=[
                    img2img_prompt_img
                ],
                outputs=[
                    img2img_prompt,
                    img2img_prompt_img
                ]
            )

            mask_mode.change(
                lambda mode, img: {
                    init_img_with_mask: gr_show(mode == 0),
                    init_img_inpaint: gr_show(mode == 1),
                    init_mask_inpaint: gr_show(mode == 1),
                },
                inputs=[mask_mode, init_img_with_mask],
                outputs=[
                    init_img_with_mask,
                    init_img_inpaint,
                    init_mask_inpaint,
                ],
            )

            img2img_args = dict(
                fn=wrap_gradio_gpu_call(modules.img2img.img2img),
                _js="submit_img2img",
                inputs=[
                           dummy_component,
                           img2img_prompt,
                           img2img_negative_prompt,
                           img2img_prompt_style,
                           img2img_prompt_style2,
                           init_img,
                           init_img_with_mask,
                           init_img_inpaint,
                           init_mask_inpaint,
                           mask_mode,
                           steps,
                           sampler_index,
                           mask_blur,
                           inpainting_fill,
                           restore_faces,
                           tiling,
                           batch_count,
                           batch_size,
                           cfg_scale,
                           denoising_strength,
                           seed,
                           subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_checkbox,
                           height,
                           width,
                           resize_mode,
                           inpaint_full_res,
                           inpaint_full_res_padding,
                           inpainting_mask_invert,
                           img2img_batch_input_dir,
                           img2img_batch_output_dir,
                       ] + custom_inputs,
                outputs=[
                    img2img_gallery,
                    generation_info,
                    html_info
                ],
                show_progress=False,
            )

            img2img_prompt.submit(**img2img_args)
            submit.click(**img2img_args)

            img2img_interrogate.click(
                fn=interrogate,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )

            if cmd_opts.deepdanbooru:
                img2img_deepbooru.click(
                    fn=interrogate_deepbooru,
                    inputs=[init_img],
                    outputs=[img2img_prompt],
                )

            roll.click(
                fn=roll_artist,
                _js="update_img2img_tokens",
                inputs=[
                    img2img_prompt,
                ],
                outputs=[
                    img2img_prompt,
                ]
            )

            prompts = [(img2img_prompt, img2img_negative_prompt)]
            style_dropdowns = [(img2img_prompt_style, img2img_prompt_style2)]
            style_js_funcs = ["update_txt2img_tokens", "update_img2img_tokens"]

            for button, (prompt, negative_prompt) in zip([img2img_save_style], prompts):
                button.click(
                    fn=add_style,
                    _js="ask_for_style_name",
                    # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
                    # the same number of parameters, but we only know the style-name after the JavaScript prompt
                    inputs=[dummy_component, prompt, negative_prompt],
                    outputs=[img2img_prompt_style, img2img_prompt_style2],
                )

            for button, (prompt, negative_prompt), (style1, style2), js_func in zip(
                    [img2img_prompt_style_apply], prompts, style_dropdowns, style_js_funcs):
                button.click(
                    fn=apply_styles,
                    _js=js_func,
                    inputs=[prompt, negative_prompt, style1, style2],
                    outputs=[prompt, negative_prompt, style1, style2],
                )

            token_button.click(fn=update_token_counter, inputs=[img2img_prompt, steps], outputs=[token_counter])

            img2img_paste_fields = [
                (img2img_prompt, "Prompt"),
                (img2img_negative_prompt, "Negative prompt"),
                (steps, "Steps"),
                (sampler_index, "Sampler"),
                (restore_faces, "Face restoration"),
                (cfg_scale, "CFG scale"),
                (seed, "Seed"),
                (width, "Size-1"),
                (height, "Size-2"),
                (batch_size, "Batch size"),
                (subseed, "Variation seed"),
                (subseed_strength, "Variation seed strength"),
                (seed_resize_from_w, "Seed resize from-1"),
                (seed_resize_from_h, "Seed resize from-2"),
                (denoising_strength, "Denoising strength"),
                *modules.scripts.scripts_img2img.infotext_fields
            ]
            parameters_copypaste.add_paste_fields("img2img", init_img, img2img_paste_fields)
            parameters_copypaste.add_paste_fields("inpaint", init_img_with_mask, img2img_paste_fields)
    return (poser, "3d poser", "3d poser"),


script_callbacks.on_ui_tabs(on_ui_tabs)
