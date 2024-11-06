
# Fine-Tune GPT-2/LLaMA Model

## Project Overview

This project focuses on fine-tuning the GPT-2/LLaMA model using the PEFT (Parameter-Efficient Fine-Tuning) and LoRA (Low-Rank Adaptation) techniques. The goal is to generate unique and engaging Twitter bios based on user input.

### Key Features

- **Model Fine-Tuning**: Fine-tunes the GPT-2/LLaMA model to generate Twitter bios based on input professions.
- **PEFT and LoRA Integration**: Utilizes PEFT and LoRA techniques to enhance the fine-tuning process.
- **Streamlit App**: Provides a user-friendly interface for generating Twitter bios using the fine-tuned model.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher
- pip and virtualenv

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/YeshwanthGovindarajan/finetuning.git
   cd finetuning
   ```

2. **Set Up Python Environment**
   ```bash
   virtualenv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

### Usage

1. **Fine-Tune the GPT-2 Model**
   ```bash
   python scripts/finetune.py
   ```

2. **Run the Streamlit App**
   ```bash
   streamlit run scripts/app.py
   ```

### Running the Tests
Ensure the system works as expected by running the tests provided.

### Contributing
Interested in contributing? We love pull requests! Here's how you can contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Create a new Pull Request.

### License
This project is licensed under the MIT License - see the LICENSE.md file for details.

### Contact
Yeshwanth G - yeshwanthgovindarjan@gmail.com
Project Link - https://github.com/YeshwanthGovindarajan/finetuning

### Acknowledgements
- Hugging Face
- Streamlit
- Python.org
