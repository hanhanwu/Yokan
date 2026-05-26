import React, { useMemo } from 'react';
import { ScrollView, StyleSheet, Text, View, Platform } from 'react-native';
import { StatusBar } from 'expo-status-bar';

// ─── Types ────────────────────────────────────────────────────────────────────

type ChunkPos = { index: number; start: number; end: number };

type ConfigData = {
  name: string;
  params: Record<string, string | number>;
  chunks: ChunkPos[];
  total: number;
};

type AllChunksData = {
  filename: string;
  text: string;
  configs: ConfigData[];
};

type Segment = {
  text: string;
  /** null = text between chunks (no highlight) */
  chunkIndex: number | null;
};

// ─── Asset (re-run experiments_goldenset/generate_chunks.py to refresh) ───────
// eslint-disable-next-line @typescript-eslint/no-var-requires
const allData: AllChunksData = require('./assets/all_chunks.json');

// ─── Palette ──────────────────────────────────────────────────────────────────

const PALETTE = [
  '#FFCDD2', '#FFE0B2', '#FFF9C4', '#DCEDC8', '#C8E6C9',
  '#B2EBF2', '#B3E5FC', '#BBDEFB', '#C5CAE9', '#D1C4E9',
  '#E1BEE7', '#F8BBD0', '#D7CCC8', '#CFD8DC', '#B2DFDB',
  '#F0F4C3', '#FFD54F', '#A5D6A7', '#80DEEA', '#CE93D8',
];

function chunkColor(index: number): string {
  return PALETTE[index % PALETTE.length];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function buildSegments(fullText: string, chunks: ChunkPos[]): Segment[] {
  const sorted = [...chunks].sort((a, b) => a.start - b.start);
  const segments: Segment[] = [];
  let cursor = 0;
  for (const chunk of sorted) {
    if (chunk.start > cursor) {
      segments.push({ text: fullText.slice(cursor, chunk.start), chunkIndex: null });
    }
    if (chunk.end > chunk.start) {
      segments.push({ text: fullText.slice(chunk.start, chunk.end), chunkIndex: chunk.index });
    }
    cursor = Math.max(cursor, chunk.end);
  }
  if (cursor < fullText.length) {
    segments.push({ text: fullText.slice(cursor), chunkIndex: null });
  }
  return segments;
}

// ─── Single config column ─────────────────────────────────────────────────────

function ConfigColumn({ cfg, text, filename }: { cfg: ConfigData; text: string; filename: string }) {
  const segments = useMemo(() => buildSegments(text, cfg.chunks), [text, cfg.chunks]);

  return (
    <View style={styles.column}>
      {/* ── Column header (sticky top) ── */}
      <View style={styles.colHeader}>
        <Text style={styles.colTitle}>{cfg.name}</Text>
        <Text style={styles.colStat}>{cfg.total} chunks</Text>

        {/* Params table */}
        <View style={styles.paramsBox}>
          {Object.entries(cfg.params).map(([k, v]) => (
            <View key={k} style={styles.paramRow}>
              <Text style={styles.paramKey}>{k}</Text>
              <Text style={styles.paramVal}>{String(v)}</Text>
            </View>
          ))}
        </View>

        {/* Chunk legend chips */}
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          style={styles.legendScroll}
          contentContainerStyle={styles.legendContent}
        >
          {cfg.chunks.map((_, i) => (
            <View key={i} style={[styles.legendChip, { backgroundColor: chunkColor(i) }]}>
              <Text style={styles.legendChipText}>{i + 1}</Text>
            </View>
          ))}
        </ScrollView>
      </View>

      {/* ── Scrollable paper area ── */}
      <ScrollView
        style={styles.paperScroll}
        contentContainerStyle={styles.paperScrollContent}
        showsVerticalScrollIndicator
        nestedScrollEnabled
      >
        <View style={styles.paper}>
          {/* Fake PDF filename bar */}
          <View style={styles.pdfHeaderRow}>
            <View style={styles.pdfLine} />
            <Text style={styles.pdfFilename} numberOfLines={1}>{filename}</Text>
            <View style={styles.pdfLine} />
          </View>

          {/* Body text with inline chunk highlights */}
          <Text style={styles.bodyText}>
            {segments.map((seg: Segment, i: number) =>
              seg.chunkIndex !== null ? (
                <Text
                  key={i}
                  style={[
                    styles.bodyText,
                    styles.highlight,
                    { backgroundColor: chunkColor(seg.chunkIndex) },
                  ]}
                >
                  {seg.text}
                </Text>
              ) : (
                <Text key={i} style={styles.bodyText}>{seg.text}</Text>
              )
            )}
          </Text>
        </View>
      </ScrollView>
    </View>
  );
}

// ─── Root component ───────────────────────────────────────────────────────────

export default function CompareChunks() {
  const { text, filename, configs } = allData;

  return (
    <View style={styles.root}>
      <StatusBar style="dark" />

      {/* Global top bar */}
      <View style={styles.topBar}>
        <Text style={styles.topTitle}>Semantic Chunk Comparison</Text>
        <Text style={styles.topSub}>
          {filename} · {configs.length} configurations
        </Text>
      </View>

      {/* Side-by-side columns — each column scrolls independently */}
      <View style={styles.columnsArea}>
        {configs.map((cfg, i) => (
          <React.Fragment key={i}>
            {i > 0 && <View style={styles.columnDivider} />}
            <ConfigColumn cfg={cfg} text={text} filename={filename} />
          </React.Fragment>
        ))}
      </View>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const IS_WEB = Platform.OS === 'web';

const styles = StyleSheet.create({
  root: {
    flex: 1,
    backgroundColor: '#E8EAED',
  },

  // ── Global top bar ──
  topBar: {
    backgroundColor: '#1A1A2E',
    paddingTop: IS_WEB ? 20 : 48,
    paddingBottom: 12,
    paddingHorizontal: 20,
    zIndex: 10,
  },
  topTitle: {
    color: '#FFFFFF',
    fontSize: 18,
    fontWeight: '700',
    letterSpacing: 0.5,
  },
  topSub: {
    color: '#A0A8C0',
    fontSize: 12,
    marginTop: 4,
  },

  // ── Columns container ──
  columnsArea: {
    flex: 1,
    flexDirection: 'row',
  },
  columnDivider: {
    width: 1,
    backgroundColor: '#C8CDD6',
  },

  // ── Single column ──
  column: {
    flex: 1,
    flexDirection: 'column',
    backgroundColor: '#E8EAED',
    minWidth: 320,
  },

  // Column header (fixed, not scrolling)
  colHeader: {
    backgroundColor: '#F8F9FA',
    borderBottomWidth: 1,
    borderBottomColor: '#DADCE0',
    paddingHorizontal: 16,
    paddingTop: 14,
    paddingBottom: 10,
  },
  colTitle: {
    fontSize: 14,
    fontWeight: '700',
    color: '#1A1A2E',
    marginBottom: 2,
  },
  colStat: {
    fontSize: 11,
    color: '#5F6368',
    marginBottom: 8,
  },

  // Params grid
  paramsBox: {
    marginBottom: 8,
    gap: 2,
  },
  paramRow: {
    flexDirection: 'row',
    gap: 6,
  },
  paramKey: {
    fontSize: 11,
    color: '#5F6368',
    fontFamily: Platform.select({ ios: 'Courier', android: 'monospace', default: 'monospace' }),
    width: 130,
  },
  paramVal: {
    fontSize: 11,
    color: '#1A1A2E',
    fontFamily: Platform.select({ ios: 'Courier', android: 'monospace', default: 'monospace' }),
    fontWeight: '600',
    flexShrink: 1,
  },

  // Legend
  legendScroll: {
    maxHeight: 32,
  },
  legendContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendChip: {
    borderRadius: 10,
    paddingHorizontal: 8,
    paddingVertical: 3,
    marginRight: 4,
  },
  legendChipText: {
    fontSize: 11,
    fontWeight: '700',
    color: '#333',
  },

  // Scrollable paper area
  paperScroll: {
    flex: 1,
  },
  paperScrollContent: {
    paddingVertical: 28,
    paddingHorizontal: 16,
    alignItems: 'center',
  },

  // White paper (PDF look)
  paper: {
    width: '100%',
    backgroundColor: '#FFFFFF',
    borderRadius: 2,
    paddingHorizontal: 40,
    paddingVertical: 48,
    ...Platform.select({
      ios: {
        shadowColor: '#000',
        shadowOffset: { width: 0, height: 3 },
        shadowOpacity: 0.15,
        shadowRadius: 10,
      },
      android: { elevation: 6 },
      web: {
        // @ts-ignore web-only
        boxShadow: '0 3px 16px rgba(0,0,0,0.15)',
      },
    }),
  },

  // PDF filename bar
  pdfHeaderRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 24,
    gap: 8,
  },
  pdfLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#C0C0C0',
  },
  pdfFilename: {
    fontSize: 9,
    color: '#999',
    letterSpacing: 0.3,
    fontFamily: Platform.select({ ios: 'Courier', android: 'monospace', default: 'monospace' }),
    flexShrink: 1,
  },

  // Body text
  bodyText: {
    fontSize: 13.5,
    lineHeight: 23,
    color: '#1A1A1A',
    fontFamily: Platform.select({
      ios: 'Georgia',
      android: 'serif',
      default: 'Georgia, "Times New Roman", serif',
    }),
  },

  // Highlighted chunk inline span
  highlight: {
    borderRadius: 2,
  },
});
