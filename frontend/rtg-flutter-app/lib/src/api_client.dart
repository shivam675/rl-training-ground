import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

/// Structured backend error: {code, message, hint}.
class ApiException implements Exception {
  ApiException(this.message, {this.code, this.hint});

  factory ApiException.fromDetail(dynamic detail, String fallback) {
    if (detail is Map) {
      return ApiException(
        detail['message']?.toString() ?? fallback,
        code: detail['code']?.toString(),
        hint: detail['hint']?.toString(),
      );
    }
    return ApiException(detail?.toString() ?? fallback);
  }

  final String message;
  final String? code;
  final String? hint;

  @override
  String toString() =>
      hint == null || hint!.isEmpty ? message : '$message\nHint: $hint';
}

class BackendApi {
  BackendApi({this.baseUrl = 'http://127.0.0.1:8000'});

  final String baseUrl;

  Uri _uri(String path) => Uri.parse('$baseUrl$path');

  Future<Map<String, dynamic>> getJson(String path) async {
    final res = await http.get(_uri(path));
    return _decode(res);
  }

  Future<Map<String, dynamic>> postJson(
    String path,
    Map<String, dynamic> body,
  ) async {
    final res = await http.post(
      _uri(path),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode(body),
    );
    return _decode(res);
  }

  Stream<Map<String, dynamic>> streamPostJson(
    String path,
    Map<String, dynamic> body,
  ) async* {
    final request = http.Request('POST', _uri(path))
      ..headers['Content-Type'] = 'application/json'
      ..body = jsonEncode(body);
    final response = await request.send();
    final lines = response.stream
        .transform(utf8.decoder)
        .transform(const LineSplitter());
    if (response.statusCode >= 400) {
      final text = await lines.join('\n');
      final fallback = '${response.statusCode} ${response.reasonPhrase}';
      try {
        final decoded = jsonDecode(text) as Map<String, dynamic>;
        throw ApiException.fromDetail(
          decoded['detail'] ?? decoded['error'],
          fallback,
        );
      } on FormatException {
        throw ApiException(text.trim().isNotEmpty ? text.trim() : fallback);
      }
    }
    await for (final line in lines) {
      if (line.trim().isEmpty) continue;
      final Map<String, dynamic> event;
      try {
        event = jsonDecode(line) as Map<String, dynamic>;
      } on FormatException {
        // Skip malformed NDJSON lines instead of killing the stream.
        continue;
      }
      yield event;
    }
  }

  Map<String, dynamic> _decode(http.Response res) {
    final text = res.body.isEmpty ? '{}' : res.body;
    final decoded = jsonDecode(text) as Map<String, dynamic>;
    if (res.statusCode >= 400) {
      throw ApiException.fromDetail(
        decoded['detail'] ?? decoded['error'],
        '${res.statusCode} ${res.reasonPhrase}',
      );
    }
    return decoded;
  }
}
