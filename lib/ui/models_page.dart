import 'package:flutter/material.dart';

import '../services/model_manager.dart';

String formatMb(int bytes) => '${(bytes / (1024 * 1024)).toStringAsFixed(0)} MB';

class ModelsPage extends StatefulWidget {
  final ModelManager manager;
  final VoidCallback onReady;

  const ModelsPage({super.key, required this.manager, required this.onReady});

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> {
  bool _running = false;
  bool _cancel = false;
  int _done = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _done = widget.manager.downloadedBytes();
  }

  Future<void> _start() async {
    setState(() {
      _running = true;
      _cancel = false;
      _error = null;
    });
    try {
      await widget.manager.downloadAll(
        onProgress: (done, total) {
          if (mounted) setState(() => _done = done);
        },
        isCancelled: () => _cancel,
      );
      if (mounted) widget.onReady();
    } on DownloadCancelled {
      if (mounted) setState(() => _error = null);
    } catch (e) {
      if (mounted) {
        setState(() => _error = 'Download interrotto: $e\nControlla la connessione e premi di nuovo "Scarica": riparte da dove si era fermato.');
      }
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
          _done = widget.manager.downloadedBytes();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final total = widget.manager.totalBytes;
    final fraction = total == 0 ? 0.0 : (_done / total).clamp(0.0, 1.0);
    return Scaffold(
      appBar: AppBar(title: const Text('Scarica i modelli')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Text(
              'Per funzionare senza Internet, l\'app ha bisogno di tre modelli di intelligenza artificiale '
              '(${formatMb(total)} in tutto). Si scaricano una volta sola: meglio usare il Wi‑Fi.',
              style: theme.textTheme.bodyLarge,
            ),
            const SizedBox(height: 16),
            Card(
              child: Column(
                children: [
                  for (final spec in ModelManager.all)
                    ListTile(
                      leading: Icon(
                        widget.manager.isReady(spec) ? Icons.check_circle : Icons.download_outlined,
                        color: widget.manager.isReady(spec) ? theme.colorScheme.primary : null,
                      ),
                      title: Text(spec.label),
                      subtitle: Text(formatMb(spec.bytes)),
                    ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            LinearProgressIndicator(value: fraction, minHeight: 8, borderRadius: BorderRadius.circular(4)),
            const SizedBox(height: 8),
            Text('${formatMb(_done)} di ${formatMb(total)}  (${(fraction * 100).toStringAsFixed(0)}%)',
                style: theme.textTheme.bodyMedium),
            if (_error != null) ...[
              const SizedBox(height: 16),
              Text(_error!, style: TextStyle(color: theme.colorScheme.error)),
            ],
            const SizedBox(height: 24),
            if (_running)
              OutlinedButton(
                onPressed: () => setState(() => _cancel = true),
                child: const Text('Metti in pausa'),
              )
            else
              FilledButton.icon(
                onPressed: _start,
                icon: const Icon(Icons.download),
                label: Text(_done > 0 ? 'Riprendi il download' : 'Scarica (${formatMb(total)})'),
              ),
            const SizedBox(height: 16),
            Text(
              'I modelli provengono dal progetto open source FaceFusion. Il modello di scambio volti '
              '(inswapper di InsightFace) è concesso solo per uso personale e non commerciale.',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
