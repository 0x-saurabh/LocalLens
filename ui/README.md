# LocalLens chat window

Flutter web app served by `locallens ui` on 127.0.0.1. It talks only to that server
(`/api/ask`, `/api/upload`, `/api/documents`, `/api/health`; see `src/locallens/server.py`).

Build once, then run `uv run locallens ui` from the repository root:

```bash
flutter build web --no-web-resources-cdn
```

- `lib/api.dart`: streaming client (newline-delimited JSON over XHR)
- `lib/chat_page.dart`: chat screen, uploads, sources panel, documents dialog
- `web/flutter_bootstrap.js`: turns off Flutter's font downloads from Google's CDN
- `fonts/`: Noto Sans (SIL Open Font License, see `fonts/OFL.txt`)
