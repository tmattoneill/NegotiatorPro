# GRADIO.md

This file provides key concepts and API usage for Gradio, the Python library used for building interactive web interfaces in this application.

## Overview

Gradio is a Python library that allows you to quickly create customizable web interfaces for machine learning models, APIs, or any Python function. It's particularly useful for creating demos, prototypes, and user-friendly interfaces for AI applications.

## Installation

```bash
pip install --upgrade gradio
```

## Core Concepts

### 1. Interface - Simple Function Wrapper

The simplest way to create a Gradio app is using `gr.Interface`:

```python
import gradio as gr

def greet(name, intensity):
    return "Hello, " + name + "!" * int(intensity)

demo = gr.Interface(
    fn=greet,
    inputs=["text", "slider"],
    outputs=["text"]
)

if __name__ == "__main__":
    demo.launch()
```

### 2. Blocks - Custom Layouts

For more complex layouts and interactions, use `gr.Blocks`:

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# My Custom Interface")
    
    with gr.Row():
        input_text = gr.Textbox(label="Input")
        output_text = gr.Textbox(label="Output")
    
    submit_btn = gr.Button("Submit")
    submit_btn.click(lambda x: x.upper(), inputs=input_text, outputs=output_text)

if __name__ == "__main__":
    demo.launch()
```

## Key Components

### Input Components
- `gr.Textbox()` - Text input
- `gr.Slider()` - Numeric slider
- `gr.Dropdown()` - Dropdown selection
- `gr.Radio()` - Radio button selection
- `gr.Checkbox()` - Checkbox input
- `gr.File()` - File upload
- `gr.Audio()` - Audio input
- `gr.Image()` - Image input

### Output Components
- `gr.Textbox()` - Text display
- `gr.Label()` - Classification labels
- `gr.Image()` - Image display
- `gr.Audio()` - Audio playback
- `gr.Video()` - Video display
- `gr.HTML()` - HTML content
- `gr.JSON()` - JSON display
- `gr.Dataframe()` - Tabular data

### Layout Components
- `gr.Row()` - Horizontal layout
- `gr.Column()` - Vertical layout
- `gr.Tab()` - Tabbed interface
- `gr.Group()` - Grouped components

## Chat Interfaces

### Basic Chat Interface

```python
import gradio as gr

def chatbot_response(message, history):
    # Your chatbot logic here
    return f"You said: {message}"

demo = gr.ChatInterface(
    chatbot_response,
    type="messages",
    title="My Chatbot"
)

if __name__ == "__main__":
    demo.launch()
```

### Advanced Chat with Message History

```python
import gradio as gr
from gradio import ChatMessage

def generate_response(history):
    # Process message history
    history.append(ChatMessage(role="user", content="Hello"))
    yield history
    
    # Generate assistant response
    history.append(ChatMessage(role="assistant", content="Hi there!"))
    yield history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(type="messages")
    button = gr.Button("Start Chat")
    button.click(generate_response, chatbot, chatbot)

if __name__ == "__main__":
    demo.launch()
```

## State Management

### Using gr.State for Session Data

```python
import gradio as gr

def increment_counter(current_count):
    return current_count + 1

with gr.Blocks() as demo:
    counter_state = gr.State(value=0)
    counter_display = gr.Number(label="Counter", value=0)
    increment_btn = gr.Button("Increment")
    
    increment_btn.click(
        increment_counter, 
        inputs=counter_state, 
        outputs=counter_state
    )
    counter_state.change(lambda x: x, counter_state, counter_display)

if __name__ == "__main__":
    demo.launch()
```

## Event Handling

### Common Events
- `.click()` - Button clicks
- `.change()` - Value changes
- `.submit()` - Form submission
- `.upload()` - File uploads
- `.stream()` - Streaming data

### Event Chaining

```python
def process_input(text):
    return text.upper()

def display_result(processed_text):
    return f"Result: {processed_text}"

# Chain events
input_box.submit(process_input, input_box, processed_state) \
         .then(display_result, processed_state, output_box)
```

## File Handling

### File Upload and Processing

```python
import gradio as gr

def process_file(file):
    if file is None:
        return "No file uploaded"
    
    # Access file properties
    filename = file.name
    content = file.read() if hasattr(file, 'read') else None
    
    return f"Processed file: {filename}"

demo = gr.Interface(
    fn=process_file,
    inputs=gr.File(label="Upload a file"),
    outputs="text"
)

if __name__ == "__main__":
    demo.launch()
```

## Advanced Features

### Custom CSS Styling

```python
css = """
.my-class {
    font-size: 20px;
    color: blue;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# Styled Interface", elem_classes=["my-class"])
```

### Authentication

```python
demo = gr.Interface(
    fn=my_function,
    inputs="text",
    outputs="text"
)

# Launch with authentication
demo.launch(auth=("username", "password"))
```

### Sharing and Deployment

```python
# Share publicly (temporary link)
demo.launch(share=True)

# Custom server settings
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    debug=True
)
```

## Integration Patterns

### With Hugging Face Models

```python
import gradio as gr
from transformers import pipeline

# Load a pre-trained model
classifier = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = classifier(text)
    return result[0]['label'], result[0]['score']

demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(label="Enter text to analyze"),
    outputs=[gr.Label(label="Sentiment"), gr.Number(label="Confidence")]
)

if __name__ == "__main__":
    demo.launch()
```

### With Custom APIs

```python
import gradio as gr
import requests

def call_api(user_input):
    response = requests.post("http://localhost:8000/api/process", 
                           json={"input": user_input})
    return response.json()["result"]

demo = gr.Interface(
    fn=call_api,
    inputs="text",
    outputs="text"
)

if __name__ == "__main__":
    demo.launch()
```

## Best Practices

### 1. Error Handling

```python
def safe_function(input_text):
    try:
        # Your processing logic
        result = process_text(input_text)
        return result
    except Exception as e:
        return f"Error: {str(e)}"
```

### 2. Progress Indicators

```python
import gradio as gr
import time

def long_running_task(text):
    for i in range(5):
        time.sleep(1)
        yield f"Processing step {i+1}/5: {text}"
    yield f"Complete: {text.upper()}"

demo = gr.Interface(
    fn=long_running_task,
    inputs="text",
    outputs="text"
)
```

### 3. Caching

```python
demo = gr.Interface(
    fn=expensive_function,
    inputs="text",
    outputs="text",
    cache_examples=True  # Cache example outputs
)
```

## Performance Considerations

- Use `yield` for streaming responses in long-running functions
- Implement proper error handling to prevent crashes
- Cache expensive computations when possible
- Use `gr.State` to maintain data across interactions
- Consider using `concurrency_limit` for resource-intensive operations

## Common Use Cases in This Application

1. **Admin Interface**: Using `gr.Blocks` with tabs for different admin functions
2. **File Upload**: Handling PDF/document uploads with `gr.File`
3. **Chat Interface**: Implementing the main negotiation guidance chat
4. **Configuration**: Managing system settings and prompts
5. **Statistics Display**: Showing usage metrics and system status

This framework provides the foundation for creating user-friendly web interfaces for AI applications without requiring frontend development expertise.