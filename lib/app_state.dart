import 'package:flutter/material.dart';

class FFAppState extends ChangeNotifier {
  static FFAppState _instance = FFAppState._internal();

  factory FFAppState() {
    return _instance;
  }

  FFAppState._internal();

  static void reset() {
    _instance = FFAppState._internal();
  }

  Future initializePersistedState() async {}

  void update(VoidCallback callback) {
    callback();
    notifyListeners();
  }

  String _loadingText = '';
  String get loadingText => _loadingText;
  set loadingText(String value) {
    _loadingText = value;
  }

  String _choiceChip = '';
  String get choiceChip => _choiceChip;
  set choiceChip(String value) {
    _choiceChip = value;
  }

  double _radius = 0.0;
  double get radius => _radius;
  set radius(double value) {
    _radius = value;
  }

  String _receivedImage = '';
  String get receivedImage => _receivedImage;
  set receivedImage(String value) {
    _receivedImage = value;
  }

  String _imageURL = '';
  String get imageURL => _imageURL;
  set imageURL(String value) {
    _imageURL = value;
  }

  String _uploadedImageByUser = '';
  String get uploadedImageByUser => _uploadedImageByUser;
  set uploadedImageByUser(String value) {
    _uploadedImageByUser = value;
  }

  bool _processStart = false;
  bool get processStart => _processStart;
  set processStart(bool value) {
    _processStart = value;
  }

  bool _APIDone = false;
  bool get APIDone => _APIDone;
  set APIDone(bool value) {
    _APIDone = value;
  }

  String _downloadImage = '';
  String get downloadImage => _downloadImage;
  set downloadImage(String value) {
    _downloadImage = value;
  }
}
