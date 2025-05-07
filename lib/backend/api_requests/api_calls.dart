import 'dart:convert';
import 'package:flutter/foundation.dart';

import '/flutter_flow/flutter_flow_util.dart';
import 'api_manager.dart';

export 'api_manager.dart' show ApiCallResponse;

const _kPrivateApiFunctionName = 'ffPrivateApiCall';

class DownscaleCall {
  static Future<ApiCallResponse> call({
    FFUploadedFile? imagePath,
  }) async {
    return ApiManager.instance.makeApiCall(
      callName: 'Downscale',
      apiUrl: 'https://jerrylapuk-e60b33d43da2.herokuapp.com/downscale',
      callType: ApiCallType.POST,
      headers: {},
      params: {
        'image': imagePath,
      },
      bodyType: BodyType.MULTIPART,
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      isStreamingApi: false,
      alwaysAllowBody: false,
    );
  }
}

class DenoiseCall {
  static Future<ApiCallResponse> call({
    FFUploadedFile? imagePath,
  }) async {
    return ApiManager.instance.makeApiCall(
      callName: 'Denoise',
      apiUrl: 'https://jerrylapuk-e60b33d43da2.herokuapp.com/denoise',
      callType: ApiCallType.POST,
      headers: {},
      params: {
        'image': imagePath,
      },
      bodyType: BodyType.MULTIPART,
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      isStreamingApi: false,
      alwaysAllowBody: false,
    );
  }
}

class UpscaleCall {
  static Future<ApiCallResponse> call({
    FFUploadedFile? imagePath,
  }) async {
    return ApiManager.instance.makeApiCall(
      callName: 'Upscale',
      apiUrl: 'https://jerrylapuk-e60b33d43da2.herokuapp.com/upscale',
      callType: ApiCallType.POST,
      headers: {},
      params: {
        'image': imagePath,
      },
      bodyType: BodyType.MULTIPART,
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      isStreamingApi: false,
      alwaysAllowBody: false,
    );
  }
}

class BlurCall {
  static Future<ApiCallResponse> call({
    FFUploadedFile? imagePath,
    int? radius = 10,
  }) async {
    return ApiManager.instance.makeApiCall(
      callName: 'Blur',
      apiUrl: 'https://jerrylapuk-e60b33d43da2.herokuapp.com/blur',
      callType: ApiCallType.POST,
      headers: {
        'Content-Type': 'application/json',
      },
      params: {
        'image': imagePath,
        'radius': radius,
      },
      bodyType: BodyType.MULTIPART,
      returnBody: true,
      encodeBodyUtf8: false,
      decodeUtf8: false,
      cache: false,
      isStreamingApi: false,
      alwaysAllowBody: false,
    );
  }
}

class ApiPagingParams {
  int nextPageNumber = 0;
  int numItems = 0;
  dynamic lastResponse;

  ApiPagingParams({
    required this.nextPageNumber,
    required this.numItems,
    required this.lastResponse,
  });

  @override
  String toString() =>
      'PagingParams(nextPageNumber: $nextPageNumber, numItems: $numItems, lastResponse: $lastResponse,)';
}

String _toEncodable(dynamic item) {
  return item;
}

String _serializeList(List? list) {
  list ??= <String>[];
  try {
    return json.encode(list, toEncodable: _toEncodable);
  } catch (_) {
    if (kDebugMode) {
      print("List serialization failed. Returning empty list.");
    }
    return '[]';
  }
}

String _serializeJson(dynamic jsonVar, [bool isList = false]) {
  jsonVar ??= (isList ? [] : {});
  try {
    return json.encode(jsonVar, toEncodable: _toEncodable);
  } catch (_) {
    if (kDebugMode) {
      print("Json serialization failed. Returning empty json.");
    }
    return isList ? '[]' : '{}';
  }
}
