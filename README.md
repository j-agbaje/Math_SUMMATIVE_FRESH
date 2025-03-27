# Math_for_Machine_Learning_Summative

Youtube video link: [https://www.youtube.com/watch?v=s0ab0CdGWMo&ab_channel=CONMEBOL&t=0s](https://www.youtube.com/watch?v=s0ab0CdGWMo&ab_channel=CONMEBOL&t=0s)

## Mission of the Project
Our mission is to improve agricultural productivity information using data-driven technological products. By leveraging a large crop production dataset, we aim to create a model that predicts crop yields based on various input features, supporting farmers, government agencies, and other stakeholders in making informed decisions.

## Brief Description of the Data
The dataset consists of over 246,000 crop production records in India, covering various states and districts. It includes features such as:
- Crop type
- Season
- Cultivation area
- Geographic location
- Crop year
- Production volume in tons

The goal is to predict crop production based on these features. This dataset provides a huge amount of information on crop production in India ranging from several years. Based on the Information the ultimate goal would be to predict crop production using powerful machine learning techniques.

## Source of Data
The dataset is publicly available on [Kaggle's Crop Production Dataset](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india) and contains valuable insights that can aid in improving agricultural planning and resource management.

## API Endpoint for Predictions
This project includes a publicly available API endpoint that returns predictions based on input values. The API is tested using Swagger UI, ensuring seamless integration and usability. You can access the API via a publicly routable URL (not localhost).

## Prerequisites

Before you begin, ensure you have the following installed:
- [Flutter SDK](https://flutter.dev/docs/get-started/install) (latest stable version)
- [Dart SDK](https://dart.dev/get-dart) (comes with Flutter)
- [Android Studio](https://developer.android.com/studio) or [Xcode](https://developer.apple.com/xcode/) (for iOS)
- [FastAPI backend](https://fastapi.tiangolo.com/) (should be already running)

## Setup Instructions

1. Clone the Repository
```bash
git clone <your-repository-url>
cd <repository-folder>
2. Navigate to the Flutter app directory
bash
Copy
cd linear_regression_model/FlutterApp/crop_production_app
3. Install Dependencies
bash
Copy
flutter pub get
4. Start your FastAPI app
bash
Copy
uvicorn main:app --reload
5. Expose your Local FastAPI App using ngrok
bash
Copy
ngrok http 8000
6. Run the app
bash
Copy
flutter run
Using the App
Fill in the required input fields

Tap the "Submit" button to send data to the FastAPI backend

Results Screen
The app will display the processed results from the FastAPI model

You can view additional details by tapping on specific result items

Troubleshooting
Connection Issues: Ensure your FastAPI backend is running and accessible

Build Errors: Run flutter clean and then flutter pub get to refresh dependencies

Platform-Specific Issues: Check Flutter's documentation for platform-specific setup

Support
For additional help, please contact [your support email] or open an issue in the repository.


