Here are the main documents related to each version of the project and their relative source codes:

- # [Report 1](./report%20on%20the%20main%20project-ver1.pdf)
    In this report, a brief introduction for [this project](../Src/main_project/) and the `RAG` is being introduced:

    

    ### **Introduction to Artificial Intelligence and Retrieval-Augmented Generation (RAG)**  
    This section provides an overview of AI and how Retrieval-Augmented Generation (RAG) is transforming the capabilities of AI-powered systems. It covers the fundamentals of RAG and how it integrates retrieval mechanisms to enhance large language models.

    

    ### **Importance of RAG**  
    Here, we explain the significance of RAG in the context of natural language processing and question-answering tasks. It explores how RAG improves accuracy by retrieving relevant information before generating responses, which is crucial for providing up-to-date and authoritative answers.

    

    ### **Differences between RAG and Other Retrieval-Generation Systems**  
    This part outlines the key differences between RAG and traditional retrieval or generation systems. It highlights how RAG bridges the gap by combining retrieval-based and generative approaches, ensuring more accurate and context-relevant outputs.

    

    ### **Role of RAG in Question Answering**  
    We introduce RAG’s specific applications in question-answering systems. This section describes how RAG enhances the accuracy of responses by pulling relevant information from external databases or knowledge sources before generating answers.

    

    ### **Structure of RAG (Retrieval-Augmented Generation)**  
    This section delves into the architecture of RAG, detailing how information is retrieved, processed, and combined with large language models to produce final responses. It provides insights into the technical aspects of RAG.

    

    ### **Explanation of the Information Retrieval Process**  
    A closer look at how the retrieval process works in RAG systems, explaining how relevant information is selected from external sources using vector searches and embedding techniques.

    

    ### **Combining Retrieval and Generation in RAG**  
    This part explains the integration of the retrieval process with generative models in RAG. It outlines the importance of prompt engineering and data augmentation in generating accurate responses.

    

    ### **Role of Large Language Models (LLMs) in Answer Generation**  
    Here, we introduce the large language models (LLMs) such as GPT and LLaMA, and discuss their importance in generating high-quality answers in RAG systems.

    

    ### **Integration and Coordination of Retrieval and Generation in RAG**  
    This section highlights the synchronization between the retrieval and generative components of RAG, ensuring smooth coordination for high-quality output.

    

    ### **Research Conducted**  
    An overview of key research studies and experiments related to RAG, discussing their findings, insights, and contributions to the advancement of the technique.

    

    ### **Comparison of Autoregressive Models (GPT Medium, GPT Large, GPT Extra Large, and LLaMA)**  
    This section presents a comparative analysis of different autoregressive models such as GPT and LLaMA, focusing on their performance in RAG pipelines and other NLP tasks.

    

    ### **Comparison of Embedding Models**  
    Here, we compare different embedding models used in RAG systems, detailing their role in improving information retrieval and how they contribute to the overall performance of RAG pipelines.

    

    ### **RAG Pipeline with Embedding and GPT**  
    This part introduces a practical implementation of the RAG pipeline using embedding models in conjunction with GPT, providing a clear understanding of how they work together to generate answers.

    

    ### **Components of the RAG Pipeline**  
    A detailed explanation of the different components that make up the RAG pipeline, highlighting how each part contributes to the retrieval and generation process.

    

    ### **Steps of the RAG Pipeline**  
    This section outlines the step-by-step process of constructing and implementing a RAG pipeline, from data preparation to final answer generation.

    

    ### **Key Considerations for Building a RAG Pipeline**  
    We provide essential factors to consider when building a RAG pipeline, such as data quality, model selection, and integration of retrieval mechanisms to optimize performance.

    

    ### **Completed Project – Bank-Related Question Conversion**  
    An introduction to a real-world project involving the conversion of bank-related questions into a RAG pipeline. This section demonstrates the practical applications of RAG in specific industries.

    

    ### **Data Processing**  
    A brief explanation of how data is processed and prepared for use in the RAG pipeline, ensuring that the input is suitable for accurate retrieval and generation.

    

    ### **Document Preparation**  
    This part focuses on how documents and other knowledge sources are prepared and indexed to enable efficient retrieval within the RAG system.

    

    ### **Model Loading**  
    Here, we explain how the selected models are loaded and integrated into the pipeline for efficient retrieval and generation tasks.

    

    ### **Building the Question-Answer Pipeline**  
    A step-by-step guide on how the question-answer pipeline is constructed using the RAG technique, from the initial question input to the final answer generation.

    
    ---

- # [Report 2](./report%20on%20second%20project-ver2.pdf)
    In this report, Here’s a brief introduction for [this project](../Models/Parsbert%20Persian%20QA.ipynb):

    

    ### **ParsBERT Model**
    This section introduces **ParsBERT**, a pre-trained language model specifically designed for Persian. It discusses the model's architecture, its applications, and how it performs in natural language processing tasks for the Persian language.

    

    ### **Data Collection**
    This part covers the process of gathering relevant Persian-language datasets, focusing on selecting high-quality and diverse data sources to train and fine-tune the model for various NLP tasks.

    

    ### **Data Preprocessing** 
    Before using the data in the model, it must go through preprocessing steps. This section outlines how the data is cleaned, tokenized, and formatted to ensure compatibility with the model.

    

    ### **Dividing Documents into Proper Persian Sentences** 
    It explains the methods used to accurately split long documents into grammatically correct Persian sentences, which is crucial for sentence-level analysis in natural language processing.

    

    ### **Implementation**  
    This section describes the steps involved in implementing the **ParsBERT** model, from data loading and integration to model configuration and deployment in a practical environment.

    

    ### **Data Preparation**
    Focuses on preparing the processed and cleaned data for model training, including encoding and creating input formats that align with the model’s requirements.

    

    ### **Model Fine-Tuning**
    Once the data is ready, the model needs to be fine-tuned for specific tasks. This section explains the methods and strategies used to adjust the pre-trained **ParsBERT** model to perform optimally on new tasks.

    

    ### **Model Evaluation**
    Finally, this part discusses the metrics and benchmarks used to evaluate the model's performance, providing insights into its accuracy and effectiveness in understanding and generating Persian text.

    
    ---

- # [Report](./report%20on%20third%20project-ver1.pdf)
    In this report, Here’s a brief introduction for [this project](../Src/Project%202/)

    

    ### **Introduction to Retrieval-Augmented Generation (RAG)**
    Retrieval-Augmented Generation (RAG) is a technique that combines the strengths of large language models (LLMs) with external knowledge bases to enhance the accuracy and relevance of generated outputs. It ensures that the model retrieves and references real-time data or domain-specific knowledge, improving its responses for specific tasks without retraining.

    

    ### **Overview of Core Technologies**
    This section will provide a summary of the core technologies used in RAG implementations. It covers key tools and frameworks essential for building RAG systems, such as language models and data retrieval systems that form the foundation of this technique.

    

    ### **Gemma Model**
    Gemma is a custom large language model (LLM) designed for specific tasks in RAG systems. This section will introduce the architecture and capabilities of the Gemma model, focusing on how it can be tailored to work with external data sources to provide accurate, contextually relevant responses.

    

    ### **Weaviate**
    Weaviate is a cloud-native vector search engine that plays a critical role in RAG systems. It helps in efficiently retrieving relevant information by performing similarity searches on vectorized data, thereby enhancing the knowledge retrieval aspect of RAG pipelines.

    

    ### **LlamaIndex**
    LlamaIndex is a specialized framework that works with large language models and data retrieval systems to enable more structured and efficient access to external knowledge sources. It is a crucial component in organizing and indexing the data used in RAG processes.

    

    ### **In-Depth Analysis of Advanced RAG**
    This section will present a detailed analysis of advanced RAG systems. It will cover how RAG can be optimized for complex queries and large-scale applications, and how it integrates external data with model-generated responses.

    

    ### **Steps to Implement Advanced RAG**
    A step-by-step guide on implementing advanced RAG systems. It outlines the technical workflow, from setting up retrieval systems to integrating LLMs and refining the output with specific knowledge bases.

    

    ### **Defining Gemma as a Custom LLM**
    This part focuses on how the Gemma model can be customized to function as a bespoke LLM for specific domains. It discusses the modifications needed to align Gemma with particular business or research needs, making it a tailored solution.

    

    ### **Integrating the Model**
    This section explains the process of integrating the custom Gemma model with the RAG system, detailing the technical considerations for merging the LLM with the retrieval engine to optimize performance.

    

    ### **Loading Data**
    Covers the process of preparing and loading external data into the system. It explains how to ensure data is compatible with both the LLM and the retrieval components of the RAG system.

    

    ### **Creating Documents with Metadata**
    This section describes the methods for structuring documents and embedding metadata, which helps in enhancing search and retrieval accuracy within the RAG framework.

    

    ### **Splitting Documents into Nodes**
    Discusses how documents are split into smaller, manageable pieces (nodes) for better retrieval and processing. This is a critical step in optimizing the RAG system’s performance.

    

    ### **Creating Indexes**
    Focuses on how to build efficient indexes for the retrieved data. Indexing is a fundamental aspect of RAG systems, as it allows for quick and relevant access to information.

    

    ### **Setting Up the Advanced RAG Query Engine**
    The final section explains how to configure and launch the query engine within the advanced RAG system. This engine allows users to query the system and receive augmented, data-driven responses from the LLM. 
