import 'package:flutter/material.dart';

import 'api.dart';
import 'chat_page.dart';

void main() {
  runApp(LocalLensApp(api: LocalLensApi()));
}

class LocalLensApp extends StatelessWidget {
  const LocalLensApp({super.key, required this.api});
  final LocalLensApi api;

  static ThemeData _theme(Brightness b) => ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2F6DB5), brightness: b),
        fontFamily: 'NotoSans',
        useMaterial3: true,
      );

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'LocalLens',
      debugShowCheckedModeBanner: false,
      theme: _theme(Brightness.light),
      darkTheme: _theme(Brightness.dark),
      home: ChatPage(api: api),
    );
  }
}
