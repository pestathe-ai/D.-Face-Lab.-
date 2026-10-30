import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:gal/gal.dart';
import 'package:image_picker/image_picker.dart';

import '../core/pipeline.dart';
import '../services/swap_engine.dart';
import 'consent_page.dart';

enum _TargetKind { photo, video }

class HomePage extends StatefulWidget {
  final SwapEngine engine;
  final Future<void> Function() onDeleteModels;
  final Future<void> Function(SwapModel) onSelectModel;

  const HomePage({super.key, required this.engine, required this.onDeleteModels, required this.onSelectModel});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _picker = ImagePicker();

  SourceFace? _source;
  _TargetKind _kind = _TargetKind.photo;
  String? _targetPath;
  String? _targetName;
  bool _allFaces = true;

  bool _busy = false;
  bool _cancel = false;
  String _stage = '';
  double? _progress;
  SwapResult? _result;
  bool _saved = false;

  void _onProgress(String stage, double? fraction) {
    if (!mounted) return;
    setState(() {
      _stage = stage;
      _progress = fraction;
    });
  }

  void _showError(Object e) {
    if (!mounted) return;
    final text = e is UserFacingError ? e.message : 'Si è verificato un errore: $e';
    showDialog<void>(
      context: context,
      builder: (c) => AlertDialog(
        title: const Text('Ops'),
        content: Text(text),
        actions: [TextButton(onPressed: () => Navigator.pop(c), child: const Text('OK'))],
      ),
    );
  }

  Future<void> _pickSource() async {
    final file = await _picker.pickImage(
      source: ImageSource.gallery,
      maxWidth: 2048,
      maxHeight: 2048,
      imageQuality: 95,
    );
    if (file == null) return;
    setState(() {
      _busy = true;
      _result = null;
    });
    try {
      final source = await widget.engine.prepareSource(file.path, _onProgress);
      setState(() => _source = source);
      if (source.facesFound > 1 && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Trovati ${source.facesFound} volti: uso il più grande.')),
        );
      }
    } catch (e) {
      _showError(e);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _pickTarget() async {
    final XFile? file;
    if (_kind == _TargetKind.photo) {
      file = await _picker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 2048,
        maxHeight: 2048,
        imageQuality: 95,
      );
    } else {
      file = await _picker.pickVideo(source: ImageSource.gallery);
    }
    if (file == null) return;
    final path = file.path;
    final name = file.name;
    setState(() {
      _targetPath = path;
      _targetName = name;
      _result = null;
    });
  }

  Future<void> _run() async {
    final source = _source, target = _targetPath;
    if (source == null || target == null) return;
    final selection = _allFaces ? FaceSelection.all : FaceSelection.largest;
    setState(() {
      _busy = true;
      _cancel = false;
      _result = null;
      _saved = false;
    });
    try {
      final result = _kind == _TargetKind.photo
          ? await widget.engine.swapPhoto(target, source, selection, _onProgress)
          : await widget.engine.swapVideo(target, source, selection, _onProgress, () => _cancel);
      setState(() => _result = result);
    } on ProcessingCancelled {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Elaborazione annullata.')));
      }
    } catch (e) {
      _showError(e);
    } finally {
      if (mounted) setState(() => _busy = false);
    }
  }

  Future<void> _save() async {
    final r = _result;
    if (r == null) return;
    try {
      if (!await Gal.hasAccess()) await Gal.requestAccess();
      if (r.isVideo) {
        await Gal.putVideo(r.path);
      } else {
        await Gal.putImage(r.path);
      }
      setState(() => _saved = true);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(const SnackBar(content: Text('Salvato nella galleria.')));
      }
    } catch (e) {
      _showError(UserFacingError('Non riesco a salvare nella galleria: $e'));
    }
  }

  Future<void> _menu(String value) async {
    if (value == 'engine') {
      final current = widget.engine.model;
      final chosen = await showDialog<SwapModel>(
        context: context,
        builder: (c) => SimpleDialog(
          title: const Text('Motore di scambio'),
          children: [
            for (final m in SwapModel.values)
              ListTile(
                leading: Icon(m == current ? Icons.radio_button_checked : Icons.radio_button_unchecked),
                title: Text(m == SwapModel.hyperswap ? 'HyperSwap (consigliato)' : 'Classico'),
                subtitle: Text(m == SwapModel.hyperswap
                    ? 'Più somigliante e dettagliato. Lavora a 256 pixel.'
                    : 'inswapper di InsightFace. Sulle foto lavora in 4 passaggi per più nitidezza (più lento).'),
                onTap: () => Navigator.pop(c, m),
              ),
          ],
        ),
      );
      if (chosen != null && chosen != current) await widget.onSelectModel(chosen);
    } else if (value == 'rules') {
      await showDialog<void>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Regole d\'uso'),
          content: SingleChildScrollView(
            child: Text(consentText.map((l) => '• $l').join('\n\n')),
          ),
          actions: [TextButton(onPressed: () => Navigator.pop(c), child: const Text('OK'))],
        ),
      );
    } else if (value == 'delete') {
      final ok = await showDialog<bool>(
        context: context,
        builder: (c) => AlertDialog(
          title: const Text('Eliminare i modelli?'),
          content: const Text('Libera circa 700 MB. Per usare di nuovo l\'app dovrai riscaricarli.'),
          actions: [
            TextButton(onPressed: () => Navigator.pop(c, false), child: const Text('Annulla')),
            FilledButton(onPressed: () => Navigator.pop(c, true), child: const Text('Elimina')),
          ],
        ),
      );
      if (ok == true) await widget.onDeleteModels();
    } else if (value == 'about') {
      showAboutDialog(
        context: context,
        applicationName: 'D. Face Lab',
        applicationVersion: '0.2.0',
        children: const [
          Text('Scambio volti sul telefono con i modelli SCRFD, ArcFace, HyperSwap (FaceFusion) e '
              'inswapper (InsightFace) distribuiti dal progetto FaceFusion. I modelli di scambio sono '
              'concessi solo per uso personale e non commerciale.'),
        ],
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final canRun = !_busy && _source != null && _targetPath != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'D. Face Lab',
          style: TextStyle(fontFamily: 'Sora', fontVariations: [FontVariation('wght', 800)]),
        ),
        actions: [
          PopupMenuButton<String>(
            onSelected: _menu,
            itemBuilder: (_) => const [
              PopupMenuItem(value: 'engine', child: Text('Motore di scambio')),
              PopupMenuItem(value: 'rules', child: Text('Regole d\'uso')),
              PopupMenuItem(value: 'delete', child: Text('Elimina modelli')),
              PopupMenuItem(value: 'about', child: Text('Informazioni')),
            ],
          ),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(16, 8, 16, 32),
          children: [
            _StepCard(
              number: 1,
              title: 'Il volto da usare',
              child: Row(
                children: [
                  _Thumb(bytes: _source?.preview, icon: Icons.face),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          _source == null
                              ? 'Per una maggiore somiglianza: foto frontale, nitida, ben illuminata, senza occhiali da sole.'
                              : 'Volto pronto.',
                          style: theme.textTheme.bodyMedium,
                        ),
                        const SizedBox(height: 8),
                        FilledButton.tonalIcon(
                          onPressed: _busy ? null : _pickSource,
                          icon: const Icon(Icons.photo_library_outlined),
                          label: Text(_source == null ? 'Scegli foto' : 'Cambia'),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            _StepCard(
              number: 2,
              title: 'Dove metterlo',
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  SegmentedButton<_TargetKind>(
                    segments: const [
                      ButtonSegment(value: _TargetKind.photo, icon: Icon(Icons.image_outlined), label: Text('Foto')),
                      ButtonSegment(value: _TargetKind.video, icon: Icon(Icons.movie_outlined), label: Text('Video')),
                    ],
                    selected: {_kind},
                    onSelectionChanged: _busy
                        ? null
                        : (s) => setState(() {
                              _kind = s.first;
                              _targetPath = null;
                              _targetName = null;
                              _result = null;
                            }),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    _targetName ??
                        (_kind == _TargetKind.photo
                            ? 'Nessuna foto scelta.'
                            : 'Nessun video scelto. Verranno elaborati al massimo i primi '
                                '${SwapEngine.maxVideoSeconds.toStringAsFixed(0)} secondi.'),
                    style: theme.textTheme.bodyMedium,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 8),
                  Align(
                    alignment: Alignment.centerLeft,
                    child: FilledButton.tonalIcon(
                      onPressed: _busy ? null : _pickTarget,
                      icon: Icon(_kind == _TargetKind.photo ? Icons.add_photo_alternate_outlined : Icons.video_library_outlined),
                      label: Text(_kind == _TargetKind.photo ? 'Scegli foto' : 'Scegli video'),
                    ),
                  ),
                  const SizedBox(height: 4),
                  SwitchListTile(
                    contentPadding: EdgeInsets.zero,
                    value: _allFaces,
                    onChanged: _busy ? null : (v) => setState(() => _allFaces = v),
                    title: const Text('Sostituisci tutti i volti'),
                    subtitle: Text(_allFaces ? 'Ogni volto trovato diventa il tuo' : 'Solo il volto più grande'),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),
            if (_busy) ...[
              LinearProgressIndicator(value: _progress, minHeight: 6, borderRadius: BorderRadius.circular(3)),
              const SizedBox(height: 8),
              Text(_stage, style: theme.textTheme.bodyMedium),
              if (_kind == _TargetKind.video && _progress != null) ...[
                const SizedBox(height: 4),
                Text('Tieni l\'app aperta e lo schermo acceso.', style: theme.textTheme.bodySmall),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerLeft,
                  child: OutlinedButton(
                    onPressed: _cancel ? null : () => setState(() => _cancel = true),
                    child: Text(_cancel ? 'Annullamento…' : 'Annulla'),
                  ),
                ),
              ],
            ] else
              FilledButton.icon(
                onPressed: canRun ? _run : null,
                icon: const Icon(Icons.auto_fix_high),
                label: const Padding(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  child: Text('Avvia'),
                ),
              ),
            if (_result != null) ...[
              const SizedBox(height: 16),
              _ResultCard(result: _result!, saved: _saved, onSave: _save),
            ],
            const SizedBox(height: 32),
            Center(
              child: Text(
                'un\'app Dom Studio',
                style: theme.textTheme.bodySmall?.copyWith(
                  fontFamily: 'Sora',
                  fontVariations: const [FontVariation('wght', 600)],
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _StepCard extends StatelessWidget {
  final int number;
  final String title;
  final Widget child;

  const _StepCard({required this.number, required this.title, required this.child});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                CircleAvatar(radius: 14, child: Text('$number')),
                const SizedBox(width: 12),
                Flexible(child: Text(title, style: theme.textTheme.titleMedium)),
              ],
            ),
            const SizedBox(height: 12),
            child,
          ],
        ),
      ),
    );
  }
}

class _Thumb extends StatelessWidget {
  final Uint8List? bytes;
  final IconData icon;

  const _Thumb({required this.bytes, required this.icon});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: Container(
        width: 84,
        height: 84,
        color: theme.colorScheme.surfaceContainerHighest,
        child: bytes == null
            ? Icon(icon, size: 40, color: theme.colorScheme.onSurfaceVariant)
            : Image.memory(bytes!, fit: BoxFit.cover, gaplessPlayback: true),
      ),
    );
  }
}

class _ResultCard extends StatelessWidget {
  final SwapResult result;
  final bool saved;
  final VoidCallback onSave;

  const _ResultCard({required this.result, required this.saved, required this.onSave});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final details = result.isVideo
        ? 'Volti sostituiti in ${result.framesWithFaces} fotogrammi su ${result.framesTotal}'
            '${result.truncated ? ' (elaborati i primi ${result.seconds.toStringAsFixed(0)} secondi)' : ''}.'
        : result.facesSwapped == 1
            ? '1 volto sostituito.'
            : '${result.facesSwapped} volti sostituiti.';
    return Card(
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Stack(
            alignment: Alignment.center,
            children: [
              Image.memory(result.preview, fit: BoxFit.contain, gaplessPlayback: true),
              if (result.isVideo)
                const CircleAvatar(radius: 28, child: Icon(Icons.play_arrow, size: 32)),
            ],
          ),
          Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(details, style: theme.textTheme.bodyMedium),
                if (result.isVideo) ...[
                  const SizedBox(height: 4),
                  Text('Qui vedi un fotogramma: il video completo lo trovi in galleria dopo averlo salvato.',
                      style: theme.textTheme.bodySmall),
                ],
                const SizedBox(height: 12),
                FilledButton.icon(
                  onPressed: saved ? null : onSave,
                  icon: Icon(saved ? Icons.check : Icons.save_alt),
                  label: Text(saved ? 'Salvato in galleria' : 'Salva in galleria'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
