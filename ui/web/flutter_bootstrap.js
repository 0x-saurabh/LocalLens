{{flutter_js}}
{{flutter_build_config}}

_flutter.loader.load({
  config: {
    // LocalLens is offline-only: never fetch fallback fonts from Google's CDN.
    // Characters missing from the bundled Noto Sans render as boxes instead.
    fontFallbackBaseUrl: "/no-font-fallback/",
  },
});
