# News Recommendation System

This project is a simple web-based News Recommendation System built using Natural Language Processing (NLP). It suggests similar news articles based on the topic entered by the user.

## Project Idea

Nowadays, there is a huge amount of news available online. It becomes difficult for users to find related or relevant news. This project helps by recommending similar news articles based on the content of a given input.

## How It Works

- The dataset contains news headlines and categories  
- Text data is cleaned by removing stopwords and unwanted characters  
- TF-IDF is used to convert text into numerical form  
- Cosine similarity is used to compare articles  
- The system returns the top 5 most similar news headlines  

## Features

- User can enter any topic (like "cricket", "politics")  
- Displays top 5 related news articles  
- Shows similarity score for each result  
- Simple and clean web interface  

## Technologies Used

- Python  
- Flask  
- Pandas  
- Scikit-learn  
- NLTK  
- HTML, CSS  

## How to Run

1. Download or clone the project  
2. Open the folder in VS Code  
3. Install required libraries:
   pip install flask pandas numpy scikit-learn nltk  

4. Run the file:
   python app.py  

5. Open browser and go to:
   http://127.0.0.1:5000/

## Limitations

- Uses only headlines (not full news content)  
- Does not consider user preferences  
- Works better with a larger dataset  

## Future Improvements

- Add full news articles instead of headlines  
- Use advanced models like Word2Vec or BERT  
- Deploy as a live website  
- Add user-based recommendations  

## Author

Rakshitha
