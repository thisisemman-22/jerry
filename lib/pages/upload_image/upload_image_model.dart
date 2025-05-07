import '/backend/api_requests/api_calls.dart';
import '/flutter_flow/flutter_flow_util.dart';
import '/flutter_flow/form_field_controller.dart';
import '/index.dart';
import 'upload_image_widget.dart' show UploadImageWidget;
import 'package:flutter/material.dart';

class UploadImageModel extends FlutterFlowModel<UploadImageWidget> {
  ///  State fields for stateful widgets in this page.

  final formKey = GlobalKey<FormState>();
  bool isDataUploading = false;
  FFUploadedFile uploadedLocalFile =
      FFUploadedFile(bytes: Uint8List.fromList([]));

  // State field(s) for ChoiceChips widget.
  FormFieldController<List<String>>? choiceChipsValueController;
  String? get choiceChipsValue =>
      choiceChipsValueController?.value?.firstOrNull;
  set choiceChipsValue(String? val) =>
      choiceChipsValueController?.value = val != null ? [val] : [];
  // State field(s) for Slider widget.
  double? sliderValue;
  // Stores action output result for [Backend Call - API (Upscale)] action in Button widget.
  ApiCallResponse? upscaleDone;
  // Stores action output result for [Backend Call - API (Downscale)] action in Button widget.
  ApiCallResponse? downscaleDone;
  // Stores action output result for [Backend Call - API (Denoise)] action in Button widget.
  ApiCallResponse? denoiseDone;
  // Stores action output result for [Backend Call - API (Blur)] action in Button widget.
  ApiCallResponse? blurDone;

  @override
  void initState(BuildContext context) {}

  @override
  void dispose() {}
}
