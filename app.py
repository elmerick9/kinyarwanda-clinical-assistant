import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartForConditionalGeneration, BartTokenizer

# Load translation model (English to Kinyarwanda using Facebook NLLB)
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
translator_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load summarization model
bart_model_name = "facebook/bart-large-cnn"
bart_tokenizer = BartTokenizer.from_pretrained(bart_model_name)
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)


def translate_text(text):
    """Translate English clinical text to Kinyarwanda."""
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True
    )
    translated_tokens = translator_model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("kin_Latn"),
        max_length=512,
    )
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


def summarize_text(text):
    """Summarize long clinical text."""
    inputs = bart_tokenizer(
        text, return_tensors="pt", max_length=1024, truncation=True
    )
    summary_ids = bart_model.generate(
        inputs["input_ids"], max_length=130, min_length=30, do_sample=False
    )
    return bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def process_text(text, task):
    """Process text based on selected task."""
    if task == "Translate to Kinyarwanda":
        return translate_text(text)
    elif task == "Summarize":
        return summarize_text(text)
    else:
        summary = summarize_text(text)
        translation = translate_text(summary)
        return f"Summary:\n{summary}\n\nKinyarwanda Translation:\n{translation}"


with gr.Blocks(title="Kinyarwanda Clinical Assistant") as demo:
    gr.Markdown("# Kinyarwanda Clinical Assistant")
    gr.Markdown(
        "AI-powered tool to translate and summarize clinical text for healthcare workers in Rwanda."
    )

    with gr.Row():
        task = gr.Radio(
            choices=["Translate to Kinyarwanda", "Summarize", "Summarize + Translate"],
            label="Select Task",
            value="Translate to Kinyarwanda",
        )

    with gr.Row():
        input_text = gr.Textbox(
            label="Enter clinical text in English",
            placeholder="Enter medical instructions, diagnosis, or clinical notes here...",
            lines=6,
        )
        output_text = gr.Textbox(label="Output", lines=6)

    submit_btn = gr.Button("Process", variant="primary")
    submit_btn.click(fn=process_text, inputs=[input_text, task], outputs=output_text)

    gr.Examples(
        examples=[
            [
                "The patient should take one tablet of paracetamol every 8 hours for 5 days.",
                "Translate to Kinyarwanda",
            ],
            [
                "The patient presents with fever, headache and body weakness for 3 days. Malaria test was positive. Treatment with artemether-lumefantrine was initiated.",
                "Summarize + Translate",
            ],
        ],
        inputs=[input_text, task],
    )

demo.launch()