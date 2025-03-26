# Math_for_Machine_Learning_Summative


Youtube video link : https://www.youtube.com/watch?v=s0ab0CdGWMo&ab_channel=CONMEBOL&t=0s

Flutter App Setup Guide
Prerequisites
Flutter SDK (latest stable version)
Dart SDK
Android Studio or VS Code
FastAPI backend deployed
Internet connection
Installation Steps
1. Clone the Repository
bash

Copy
git clone https://github.com/yourusername/your-app-repo.git
cd your-app-repo
2. Install Dependencies
bash

Copy
flutter pub get
3. Configure API Connection
Create a .env file in the project root:


Copy
API_BASE_URL=https://your-fastapi-backend.com/api
4. Environment Setup
Ensure you have the following packages in pubspec.yaml:

http: ^0.13.5
flutter_dotenv: ^5.0.2
video_player: ^2.6.1
5. Run the Application
bash

Copy
flutter run
Video Demo
Show Image

Troubleshooting
Ensure FastAPI backend is running
Check network connectivity
Verify API endpoint in .env file
API Interaction
The app sends inputs to the FastAPI backend
Receives and displays outputs from the model
Handles potential network and parsing errors
Supported Platforms
Android
iOS
Web
Contributing
Fork the repository
Create your feature branch
Commit changes
Push to the branch
Create a Pull Request
