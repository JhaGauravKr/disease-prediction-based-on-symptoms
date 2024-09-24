##Overview
This project is a machine learning-based application designed to predict diseases based on a user's selected symptoms. The prediction is made using four different algorithms: **Decision Tree, Random Forest, K-Nearest Neighbour, and Naive Bayes**. The user interacts with the application through a simple Tkinter GUI, and the prediction results are stored in an **SQLite database**.

###Features
**Multiple ML algorithms:** Uses Decision Tree, Random Forest, K-Nearest Neighbour, and Naive Bayes to provide diverse predictions.
**User-friendly GUI:** Built with Tkinter, the interface allows users to select symptoms and view predictions easily.
**Final prediction:** After viewing results from all algorithms, users can click the "Final Prediction" button to get a consolidated prediction.
**Database storage:** All user data, including patient names and symptoms, is stored in an SQLite database.
###Tech Stack
**Programming Language:** Python
**GUI:** Tkinter
**Machine Learning Algorithms:**
    - Decision Tree
    - Random Forest
    - K-Nearest Neighbour
    - Naive Bayes
**Database:** SQLite
**Development Environment:** Jupyter Notebook
###How It Works
**Input:** The user provides their name and selects five symptoms from a predefined list.
**Prediction:** Upon clicking the "Predict All" button, the application uses four machine learning models to predict potential diseases based on the symptoms.
**Final Outcome:** The user can then click the "Final Prediction" button to see a consolidated prediction based on the results of the four algorithms.
**Database Storage:** The patient's details and predictions are saved in an SQLite database, which can be accessed through the SQLite browser for further analysis.


| Algorithm            | Accuracy |
|----------------------|----------|
| Decision Tree        | 93%      |
| Random Forest        | 95%      |
| K-Nearest Neighbour  | 92%      |
| Naive Bayes          | 93%      |

###Future Improvements
1. Adding more symptoms and diseases to enhance the prediction model.
2. Incorporating more sophisticated ML models like SVM or Neural Networks for better accuracy.
3. Implementing a web-based interface using Django/Flask for broader accessibility.

###Conclusion
This project demonstrates the integration of machine learning with a user-friendly interface for disease prediction. The use of multiple algorithms offers varied insights, making the final outcome more reliable.

###License
This project is licensed under the MIT License - see the LICENSE file for details.
