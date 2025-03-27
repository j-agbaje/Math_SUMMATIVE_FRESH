import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'output_page.dart';

void main() {
  runApp(CropProductionApp());
}

class CropProductionApp extends StatelessWidget {
  const CropProductionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      theme: ThemeData(
        primarySwatch: Colors.green,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: const InputPage(),
    );
  }
}

class InputPage extends StatefulWidget {
  const InputPage({super.key});

  @override
  State<InputPage> createState() => _InputPageState();
}

class _InputPageState extends State<InputPage> {
  final _formKey = GlobalKey<FormState>();

  // Controllers for each input field
  final TextEditingController _areaController = TextEditingController();
  final TextEditingController _cropYearController = TextEditingController();
  final TextEditingController _stateNameController = TextEditingController();
  final TextEditingController _districtNameController = TextEditingController();
  final TextEditingController _seasonController = TextEditingController();
  final TextEditingController _cropController = TextEditingController();

  // List of valid states
  final List<String> _validStates = [
    'Andaman and Nicobar Islands',
    'Andhra Pradesh',
    'Arunachal Pradesh',
    'Assam',
    'Bihar',
    'Chandigarh',
    'Chhattisgarh',
    'Dadra and Nagar Haveli',
    'Goa',
    'Gujarat',
    'Haryana',
    'Himachal Pradesh',
    'Jammu and Kashmir',
    'Jharkhand',
    'Karnataka',
    'Kerala',
    'Madhya Pradesh',
    'Maharashtra',
    'Manipur',
    'Meghalaya',
    'Mizoram',
    'Nagaland',
    'Odisha',
    'Puducherry',
    'Punjab',
    'Rajasthan',
    'Sikkim',
    'Tamil Nadu',
    'Telangana',
    'Tripura',
    'Uttar Pradesh',
    'Uttarakhand',
    'West Bengal',
  ];

  // List of valid seasons
  final List<String> _validSeasons = [
    'Kharif',
    'Whole year',
    'Autumn',
    'Rabi',
    'Summer',
    'Winter',
  ];

  // List of valid crops
  final List<String> _validCrops = [
    'Rice',
    'Wheat',
    'Maize',
    'Sugarcane',
    'Cotton(lint)',
    'Potato',
    'Onion',
    'Tomato',
    'Banana',
    'Orange',
    'Coconut',
    'Groundnut',
    'Soyabean',
    'Arhar/Tur',
    'Gram',
    'Jowar',
    'Bajra',
    'Ragi',
  ];

  Future<void> _predictCropProduction() async {
    if (_formKey.currentState!.validate()) {
      final url = Uri.parse(
        'https://dbbb-2c0f-eb68-627-400-a5b1-7978-2ba4-dd67.ngrok-free.app/predict',
      );

      final payload = {
        "Area": double.parse(_areaController.text),
        "Crop_Year": int.parse(_cropYearController.text),
        "State_Name": _stateNameController.text,
        "District_Name": _districtNameController.text.toUpperCase(),
        "Season": _seasonController.text,
        "Crop": _cropController.text,
      };

      try {
        final response = await http.post(
          url,
          headers: {"Content-Type": "application/json"},
          body: json.encode(payload),
        );

        if (response.statusCode == 200) {
          final responseData = json.decode(response.body);

          // Navigate to output page
          Navigator.push(
            context,
            MaterialPageRoute(
              builder:
                  (context) => OutputPage(
                    predictedProduction: responseData['predicted_production'],
                    inputData: payload,
                  ),
            ),
          );
        } else {
          // Show error dialog
          _showErrorDialog('Prediction failed: ${response.body}');
        }
      } catch (e) {
        _showErrorDialog('Error connecting to server: $e');
      }
    }
  }

  void _showErrorDialog(String message) {
    showDialog(
      context: context,
      builder:
          (ctx) => AlertDialog(
            title: const Text('Error'),
            content: Text(message),
            actions: [
              TextButton(
                child: const Text('Okay'),
                onPressed: () {
                  Navigator.of(ctx).pop();
                },
              ),
            ],
          ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Crop Production Predictor'),
        backgroundColor: Colors.green,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Area Input
              TextFormField(
                controller: _areaController,
                decoration: const InputDecoration(
                  labelText: 'Area (hectares)',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter area';
                  }
                  final area = double.tryParse(value);
                  if (area == null || area <= 0 || area > 8580100) {
                    return 'Enter a valid area between 0 and 8,580,100';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 10),

              // Crop Year Input
              TextFormField(
                controller: _cropYearController,
                decoration: const InputDecoration(
                  labelText: 'Crop Year',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter crop year';
                  }
                  final year = int.tryParse(value);
                  if (year == null || year < 1996 || year > 2035) {
                    return 'Enter a valid year between 1996 and 2035';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 10),

              // State Name Input
              TextFormField(
                controller: _stateNameController,
                decoration: const InputDecoration(
                  labelText: 'State Name',
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter state name';
                  }
                  if (!_validStates.contains(value.trim())) {
                    return 'Enter a valid Indian state';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 10),

              // District Name Input
              TextFormField(
                controller: _districtNameController,
                decoration: const InputDecoration(
                  labelText: 'District Name',
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter district name';
                  }
                  if (!RegExp(r'^[a-zA-Z\s]+$').hasMatch(value)) {
                    return 'District name must contain only alphabets';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 10),

              // Season Input
              TextFormField(
                controller: _seasonController,
                decoration: const InputDecoration(
                  labelText: 'Season',
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter season';
                  }
                  if (!_validSeasons.contains(value.trim())) {
                    return 'Enter a valid season';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 10),

              // Crop Input
              TextFormField(
                controller: _cropController,
                decoration: const InputDecoration(
                  labelText: 'Crop',
                  border: OutlineInputBorder(),
                ),
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter crop name';
                  }
                  if (!_validCrops.contains(value.trim())) {
                    return 'Enter a valid crop';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 20),

              // Predict Button
              ElevatedButton(
                onPressed: _predictCropProduction,
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.green,
                  padding: const EdgeInsets.symmetric(vertical: 15),
                ),
                child: const Text('Predict', style: TextStyle(fontSize: 18)),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

