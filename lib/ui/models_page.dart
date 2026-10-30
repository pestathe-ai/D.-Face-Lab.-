import 'package:flutter/material.dart';

import '../core/pipeline.dart';
import '../services/model_manager.dart';

String formatMb(int bytes) => '${(bytes / (1024 * 1024)).toStringAsFixed(0)} MB';

String modelName(SwapModel m) => m == SwapModel.hyperswap ? 'HyperSwap' : 'Classico';

class ModelsPage extends StatefulWidget {
  final ModelManager manager;
  final VoidCallback onReady;
  final Future<void> Function(SwapModel) onSelectModel;

  const ModelsPage({super.key, required this.manager, required this.onReady, required this.onSelectModel});

  @override
  State<ModelsPage> createState() => _ModelsPageState();
}

class _ModelsPageState extends State<ModelsPage> {
  bool _running = false;
  bool _cancel = false;
  int _done = 0;
  String? _error;

  SwapModel get _model => widget.manager.preferredModel;

  @override
  void initState() {
    super.initState();
    _done = widget.manager.downloadedBytes(_model);
  }

  Future<void> _start() async {
    setState(() {
      _running = true;
      _cancel = false;
      _error = null;
    });
    try {
      await widget.manager.downloadFor(
        _model,
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
        setState(() => _error = 'Download interrotto: $e\nControlla la connessione e premi di nuovo: riparte da dove si era fermato.');
      }
    } finally {
      if (mounted) {
        setState(() {
          _running = false;
          _done = widget.manager.downloadedBytes(_model);
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final m = widget.manager;
    final total = m.totalBytes(_model);
    final missing = m.missingBytes(_model);
    final fraction = total == 0 ? 0.0 : (_done / total).clamp(0.0, 1.0);
    final upgrading = _model == SwapModel.hyperswap && m.readyFor(SwapModel.inswapper);
    final other = _model == SwapModel.hyperswap ? SwapModel.inswapper : SwapModel.hyperswap;
    return Scaffold(
      appBar: AppBar(title: Text(upgrading ? 'Nuovo motore HyperSwap' : 'Scarica i modelli')),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.all(24),
          children: [
            Text(
              upgrading
                  ? 'HyperSwap lavora a una risoluzione doppia e trasferisce meglio i tratti della persona: '
                      'il risultato somiglia di più. Serve un download in più di ${formatMb(missing)}.'
                  : 'Per funzionare senza Internet, l\'app ha bisogno di alcuni modelli di intelligenza artificiale '
                      '(${formatMb(missing)} da scaricare). Si scaricano una volta sola: meglio usare il Wi‑Fi.',
              style: theme.textTheme.bodyLarge,
            ),
            const SizedBox(height: 16),
            Card(
              child: Column(
                children: [
                  for (final spec in ModelManager.requiredFor(_model))
                    ListTile(
                      leading: Icon(
                        m.isReady(spec) ? Icons.check_circle : Icons.download_outlined,
                        color: m.isReady(spec) ? theme.colorScheme.primary : null,
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
            else ...[
              FilledButton.icon(
                onPressed: _start,
                icon: const Icon(Icons.download),
                label: Text(_done > 0 && _done < total ? 'Riprendi il download' : 'Scarica (${formatMb(missing)})'),
              ),
              const SizedBox(height: 8),
              TextButton(
                onPressed: () => widget.onSelectModel(other),
                child: Text(m.readyFor(other)
                    ? 'Per ora continua con il motore ${modelName(other)}'
                    : 'Usa invece il motore ${modelName(other)} (${formatMb(m.missingBytes(other))})'),
              ),
            ],
            const SizedBox(height: 16),
            Text(
              'I modelli provengono dal progetto open source FaceFusion. HyperSwap (FaceFusion) e '
              'inswapper (InsightFace) sono concessi solo per uso personale e non commerciale.',
              style: theme.textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}
