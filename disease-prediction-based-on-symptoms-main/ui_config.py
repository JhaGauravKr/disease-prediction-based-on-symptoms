import streamlit as st

def set_page_config_and_styles():
    """
    Sets up the Streamlit page configuration and applies custom CSS styles.
    """
    st.set_page_config(
        page_title="Smart Disease Predictor",
        page_icon="💊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        .main-header {
            font-size: 3.5em;
            color: #FF4B4B;
            text-align: center;
            font-family: 'Times New Roman', serif;
            font-style: italic;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .subheader {
            font-size: 1.8em;
            color: #1E90FF;
            text-align: center;
            font-family: 'Times New Roman', serif;
            font-style: italic;
            font-weight: bold;
            margin-bottom: 1.5em;
        }
        .sidebar .sidebar-content {
            background-color: #f0f2f6; /* Light gray background for sidebar */
        }
        .stSelectbox, .stTextInput, .stButton > button {
            border-radius: 8px;
            border: 1px solid #FF4B4B; /* Red accent for inputs */
        }
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
            font-weight: bold;
            padding: 0.7em 1.5em;
            transition: background-color 0.3s;
        }
        .stButton > button:hover {
            background-color: #FF0000;
            border-color: #FF0000;
        }
        .stAlert {
            border-radius: 8px;
        }
        .stPlotlyChart {
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            padding: 1em;
            background-color: white;
        }
        .result-box {
            border: 2px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 10px;
            background-color: #f9f9f9;
            font-size: 1.1em;
        }
        .result-box.success {
            border-color: #28a745;
            background-color: #e6ffed;
        }
        .result-box.info {
            border-color: #007bff;
            background-color: #e0f2ff;
        }
        .result-box.warning {
            border-color: #ffc107;
            background-color: #fff8e0;
        }
        /* Specific styles for prediction result backgrounds and text colors */
        .prediction-label {
            font-family: "Times", serif;
            font-weight: bold;
            font-style: italic;
            font-size: 15px; /* Matches the original label size */
            padding: 8px; /* Some padding for better look */
            border-radius: 5px; /* Slightly rounded corners */
            text-align: center; /* Center the text */
            width: 100%; /* Take full width of its container column */
            box-sizing: border-box; /* Include padding in width */
        }
        .bg-dt { background-color: #6C5B7B; color: white; } /* Darker purple, white text */
        .bg-rf { background-color: #C06C84; color: white; } /* Muted pink, white text */
        .bg-nb { background-color: #F67280; color: white; } /* Coral, white text */
        .bg-knn { background-color: #FFB3A7; color: black; } /* Lighter peach, black text */
        .bg-final { background-color: #4CAF50; color: white; } /* Green, white text */
        .bg-final-warning { background-color: #FFC107; color: black; } /* Orange, black text */

        </style>
        """, unsafe_allow_html=True)