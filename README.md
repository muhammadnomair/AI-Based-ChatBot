# AI-Based-ChatBot
This repository contains a chatbot that enables users to upload files as input, which then uses them to train itself. The chatbot parses the data from the file and generates its responses on the content (information) within the file, providing customized answers tailored to the information it has been trained on.

### Key Features:

**File Upload Training:** The chatbot accepts various file formats (e.g., text, CSV, PDF) and extracts relevant data to learn from.

**Dynamic Responses:** After processing the file, the chatbot generates responses specific to the uploaded data, offering accurate and contextual answers.

**Easy Integration:** Designed to be integrated with web applications, this chatbot can be easily added to existing platforms like WordPress or standalone PHP-based sites.

 **Customizable & Extensible:** The chatbot’s architecture allows for easy modifications, enabling it to handle specific domains of knowledge depending on the input file.

### How to Use this Project

**Set-up the Environment**

First of all, the user need to create a seperate virtual environment (venv), where all the dependeccies libraries should be installed.These depencies are listed in the requirements file.

**Upload File to Communicate With**

You can upload any desried file, such as text, CSV, PDF to perform conversation (questions and answers) with by passing the file-path, where it is located.

**Parsing, indexing, and embedding the Uploaded Files**

The implementation of these critical proces has been already done in this proejct. Please refer to the fastapi_app.py file for more details.

**Get Response from ChatBot**

After it has been trained on the uploaded files, you can call `answer_with_rag()`method, in the fastapi_app.py file, to get response based on the provided question.

**Deploy and Test**

You can deploy this chatbot with a web interface using Flask, Streamlit, or other interfaces. The whole project code has been configured and optimized according to the Fastapi architecture. So, the user can adopt these end-point properly.

Best of Luck!
