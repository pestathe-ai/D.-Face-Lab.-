import 'dart:io';

import 'package:flutter/material.dart';

import 'services/model_manager.dart';
import 'services/swap_engine.dart';
import 'ui/consent_page.dart';
import 'ui/home_page.dart';
import 'ui/models_page.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const FaceLabApp());
}

class FaceLabApp extends StatelessWidget {
  const FaceLabApp({super.key});

  @override
  Widget build(BuildContext context) {
    const seed = Color(0xFF6B4EFF);
    return MaterialApp(
      title: 'D. Face Lab',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.light, useMaterial3: true),
      darkTheme: ThemeData(colorSchemeSeed: seed, brightness: Brightness.dark, useMaterial3: true),
      home: const AppRoot(),
    );
  }
}

enum _Phase { loading, consent, models, home }

class AppRoot extends StatefulWidget {
  const AppRoot({super.key});

  @override
  State<AppRoot> createState() => _AppRootState();
}

class _AppRootState extends State<AppRoot> {
  _Phase _phase = _Phase.loading;
  ModelManager? _manager;
  SwapEngine? _engine;

  File _consentFile(ModelManager m) => File('${m.dir.parent.path}/consent_v1');

  @override
  void initState() {
    super.initState();
    _start();
  }

  Future<void> _start() async {
    final manager = await ModelManager.create();
    _manager = manager;
    _route();
  }

  void _route() {
    final m = _manager!;
    setState(() {
      if (!_consentFile(m).existsSync()) {
        _phase = _Phase.consent;
      } else if (!m.allReady) {
        _phase = _Phase.models;
      } else {
        _engine ??= SwapEngine(m);
        _phase = _Phase.home;
      }
    });
  }

  Future<void> _accept() async {
    await _consentFile(_manager!).writeAsString(DateTime.now().toIso8601String());
    _route();
  }

  Future<void> _deleteModels() async {
    await _engine?.dispose();
    _engine = null;
    await _manager!.deleteAll();
    _route();
  }

  @override
  void dispose() {
    _engine?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    switch (_phase) {
      case _Phase.loading:
        return const Scaffold(body: Center(child: CircularProgressIndicator()));
      case _Phase.consent:
        return ConsentPage(onAccept: _accept);
      case _Phase.models:
        return ModelsPage(manager: _manager!, onReady: _route);
      case _Phase.home:
        return HomePage(engine: _engine!, onDeleteModels: _deleteModels);
    }
  }
}
