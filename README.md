### 🤖 Recipe Bot

This recipe bot is our natural language processing (NLP) course project. When you input ingredients (e.g., 'I have toast, cream, and cinnamon'), the bot will provide a recommended recipe, including the recipe title, ingredients, and instructions based on the input.

### Overview

**Dataset:** https://github.com/rtlee9/recipe-box

**Data Cleaning**
- Remove the word "Advertisement" from the end of every instruction.
- Remove any null (missing) data entries.
- Remove duplicated data entries to ensure data integrity.

**Data Pre-processing**
- Lemmatization for regular expressions
- SpaCy for extracting cooking time details from the instruction column.

**Model Building**
- TF-IDF Vectorization
- Word2Vec

**Chatbot Interface**
- Streamlit
