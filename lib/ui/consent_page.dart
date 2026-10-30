import 'package:flutter/material.dart';

const consentText = [
  'Usa solo volti di persone che ti hanno dato il permesso, compreso il tuo.',
  'Non creare contenuti per ingannare, truffare, diffamare o molestare qualcuno.',
  'Non creare contenuti sessuali o umilianti con il volto di altre persone, né contenuti con minori.',
  'Se condividi un risultato, indica chiaramente che è un\'immagine modificata.',
];

class ConsentPage extends StatefulWidget {
  final VoidCallback onAccept;

  const ConsentPage({super.key, required this.onAccept});

  @override
  State<ConsentPage> createState() => _ConsentPageState();
}

class _ConsentPageState extends State<ConsentPage> {
  bool _checked = false;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Scaffold(
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(24, 32, 24, 24),
          children: [
            Icon(Icons.face_retouching_natural, size: 56, color: theme.colorScheme.primary),
            const SizedBox(height: 16),
            Text('Benvenuto in D. Face Lab', style: theme.textTheme.headlineSmall),
            const SizedBox(height: 8),
            Text(
              'L\'app sostituisce un volto in foto e video direttamente sul telefono: '
              'le tue immagini non vengono inviate a nessun server.',
              style: theme.textTheme.bodyLarge,
            ),
            const SizedBox(height: 24),
            Text('Prima di iniziare', style: theme.textTheme.titleMedium),
            const SizedBox(height: 8),
            for (final line in consentText)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 6),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.check_circle_outline, size: 20, color: theme.colorScheme.primary),
                    const SizedBox(width: 12),
                    Expanded(child: Text(line, style: theme.textTheme.bodyMedium)),
                  ],
                ),
              ),
            const SizedBox(height: 8),
            Text(
              'Creare o diffondere immagini false di una persona senza il suo consenso può essere un reato.',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 16),
            CheckboxListTile(
              value: _checked,
              onChanged: (v) => setState(() => _checked = v ?? false),
              contentPadding: EdgeInsets.zero,
              controlAffinity: ListTileControlAffinity.leading,
              title: const Text('Ho letto e mi impegno a rispettare queste regole'),
            ),
            const SizedBox(height: 16),
            FilledButton(
              onPressed: _checked ? widget.onAccept : null,
              child: const Padding(
                padding: EdgeInsets.symmetric(vertical: 12),
                child: Text('Continua'),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
