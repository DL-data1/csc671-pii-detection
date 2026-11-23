import gradio as gr

from predict import predict, highlight_text, redact_text


MODEL_PATH = "models/distilbert-small"


def run_detection(text):
    entities, processed, total = predict(MODEL_PATH, text)

    json_entities = []

    for entity in entities:
        json_entities.append({
            "type": entity["type"],
            "text": text[entity["start"]:entity["end"]],
            "start": entity["start"],
            "end": entity["end"]
        })

    highlighted = highlight_text(text, entities)
    redacted = redact_text(text, entities)

    return highlighted, json_entities, redacted


with gr.Blocks(title="PII Detection Demo") as demo:
    gr.Markdown("# PII Detection Demo")
    gr.Markdown(
        "Enter text below to detect personally identifiable information."
    )

    input_text = gr.Textbox(
        label="Input text",
        lines=6,
        placeholder="My name is John Smith and my email is john@example.com."
    )

    detect_button = gr.Button("Detect PII")

    highlighted_output = gr.Textbox(
        label="Highlighted PII",
        lines=6
    )

    json_output = gr.JSON(
        label="JSON Output"
    )

    redacted_output = gr.Textbox(
        label="Redacted Text",
        lines=6
    )

    detect_button.click(
        fn=run_detection,
        inputs=input_text,
        outputs=[
            highlighted_output,
            json_output,
            redacted_output
        ]
    )


if __name__ == "__main__":
    demo.launch()