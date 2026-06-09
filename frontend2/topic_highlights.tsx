import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { StatusBar } from 'expo-status-bar';

// ─── Types ────────────────────────────────────────────────────────────────────

type TopicMatch = {
  rank: number;
  rrf_score: number;
  semantic_score: number;
  bm25_score: number;
  doc_name: string;
  chunk_index: number;
  start: number;
  end: number;
  matched_text: string;
};

type RetrievalData = {
  filename: string;
  text: string;
  topics: string[];
  results: Record<string, TopicMatch[]>;
};

type Segment = {
  text: string;
  topicIndex: number | null; // null = unhighlighted gap
  matchRank: number;
};

// ─── Asset ───────────────────────────────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-var-requires
const data: RetrievalData = require('./assets/topic_retrieval.json');

// ─── Topic colour palette (solid accent colours) ─────────────────────────────
const TOPIC_COLORS = [
  { bg: '#FFCDD2', border: '#E57373', text: '#B71C1C' }, // red
  { bg: '#C8E6C9', border: '#66BB6A', text: '#1B5E20' }, // green
  { bg: '#BBDEFB', border: '#42A5F5', text: '#0D47A1' }, // blue
  { bg: '#FFF9C4', border: '#FFEE58', text: '#F57F17' }, // yellow
  { bg: '#E1BEE7', border: '#AB47BC', text: '#4A148C' }, // purple
  { bg: '#B2EBF2', border: '#26C6DA', text: '#006064' }, // cyan
  { bg: '#FFE0B2', border: '#FFA726', text: '#BF360C' }, // orange
  { bg: '#F8BBD0', border: '#EC407A', text: '#880E4F' }, // pink
];

function topicColor(index: number) {
  return TOPIC_COLORS[index % TOPIC_COLORS.length];
}

type ActiveMatch = { topic: string; rank: number } | null;

// ─── Build document segments ──────────────────────────────────────────────────
// When activeMatch is set, highlight only that one span.
// When null, highlight all spans (highest-rrf wins on overlap).

function buildSegments(
  fullText: string,
  topics: string[],
  results: Record<string, TopicMatch[]>,
  activeMatch: ActiveMatch,
): Segment[] {
  if (activeMatch !== null) {
    const matches = results[activeMatch.topic] ?? [];
    const match = matches.find((m) => m.rank === activeMatch.rank);
    if (!match) return [{ text: fullText, topicIndex: null, matchRank: 0 }];
    const topicIndex = topics.indexOf(activeMatch.topic);
    const segs: Segment[] = [];
    if (match.start > 0) segs.push({ text: fullText.slice(0, match.start), topicIndex: null, matchRank: 0 });
    segs.push({ text: fullText.slice(match.start, match.end), topicIndex, matchRank: match.rank });
    if (match.end < fullText.length) segs.push({ text: fullText.slice(match.end), topicIndex: null, matchRank: 0 });
    return segs;
  }

  // No selection — show all highlights
  const spanMap: Map<string, { topicIndex: number; rank: number; end: number }> = new Map();
  topics.forEach((topic, tIdx) => {
    const matches = results[topic] ?? [];
    matches.forEach((m) => {
      const key = `${m.start}`;
      const existing = spanMap.get(key);
      if (!existing || m.rank < existing.rank || (m.rank === existing.rank && tIdx < existing.topicIndex)) {
        spanMap.set(key, { topicIndex: tIdx, rank: m.rank, end: m.end });
      }
    });
  });

  const intervals = Array.from(spanMap.entries())
    .map(([startStr, v]) => ({ start: parseInt(startStr, 10), end: v.end, topicIndex: v.topicIndex, rank: v.rank }))
    .sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;
  for (const span of intervals) {
    if (span.start < cursor) continue;
    if (span.start > cursor) segments.push({ text: fullText.slice(cursor, span.start), topicIndex: null, matchRank: 0 });
    segments.push({ text: fullText.slice(span.start, span.end), topicIndex: span.topicIndex, matchRank: span.rank });
    cursor = span.end;
  }
  if (cursor < fullText.length) segments.push({ text: fullText.slice(cursor), topicIndex: null, matchRank: 0 });
  return segments;
}

// ─── Score bar (mini) ─────────────────────────────────────────────────────────

function ScoreBar({ label, value, max, color }: { label: string; value: number; max: number; color: string }) {
  const pct = Math.min(value / max, 1) * 100;
  return (
    <View style={scoreBarStyles.row}>
      <Text style={scoreBarStyles.label}>{label}</Text>
      <View style={scoreBarStyles.track}>
        <View style={[scoreBarStyles.fill, { width: `${pct}%` as any, backgroundColor: color }]} />
      </View>
      <Text style={scoreBarStyles.val}>{value.toFixed(4)}</Text>
    </View>
  );
}

const scoreBarStyles = StyleSheet.create({
  row: { flexDirection: 'row', alignItems: 'center', marginBottom: 3 },
  label: { fontSize: 10, color: '#666', width: 38, fontFamily: 'monospace' },
  track: { flex: 1, height: 6, backgroundColor: '#E0E0E0', borderRadius: 3, marginHorizontal: 4 },
  fill: { height: 6, borderRadius: 3 },
  val: { fontSize: 10, color: '#333', width: 44, textAlign: 'right', fontFamily: 'monospace' },
});

// ─── Topic side-panel card ────────────────────────────────────────────────────

function TopicCard({
  topic,
  topicIndex,
  matches,
  activeRank,
  onMatchSelect,
}: {
  topic: string;
  topicIndex: number;
  matches: TopicMatch[];
  activeRank: number | null;
  onMatchSelect: (rank: number) => void;
}) {
  const color = topicColor(topicIndex);

  return (
    <View style={topicCardStyles.card}>
      {/* Header row */}
      <View style={topicCardStyles.header}>
        <View style={[topicCardStyles.dot, { backgroundColor: color.border }]} />
        <Text style={[topicCardStyles.topicName, { color: color.text }]} numberOfLines={2}>
          {topic}
        </Text>
        <View style={[topicCardStyles.badge, { backgroundColor: color.bg }]}>
          <Text style={[topicCardStyles.badgeText, { color: color.text }]}>{matches.length}</Text>
        </View>
      </View>

      {/* Clickable rank buttons */}
      <View style={topicCardStyles.rankRow}>
        {matches.map((m) => {
          const isSelected = activeRank === m.rank;
          return (
            <TouchableOpacity
              key={m.chunk_index}
              style={[
                topicCardStyles.rankBtn,
                { borderColor: color.border, backgroundColor: isSelected ? color.border : color.bg },
              ]}
              onPress={() => onMatchSelect(m.rank)}
              activeOpacity={0.7}
            >
              <Text style={[topicCardStyles.rankBtnText, { color: isSelected ? '#fff' : color.text }]}>
                #{m.rank}
              </Text>
            </TouchableOpacity>
          );
        })}
      </View>

      {/* Score bars for all matches */}
      <View style={topicCardStyles.scoreSection}>
        {matches.map((m) => {
          const isSelected = activeRank === m.rank;
          return (
            <View
              key={m.chunk_index}
              style={[
                topicCardStyles.matchScoreBlock,
                isSelected && { backgroundColor: color.bg, borderRadius: 4 },
              ]}
            >
              <Text style={[topicCardStyles.matchScoreLabel, { color: color.text }]}>#{m.rank}</Text>
              <ScoreBar label="rrf"  value={m.rrf_score}       max={0.02} color={color.border} />
              <ScoreBar label="sem"  value={m.semantic_score}  max={1}    color={color.border} />
              <ScoreBar label="bm25" value={m.bm25_score}      max={20}   color={color.border} />
            </View>
          );
        })}
      </View>
    </View>
  );
}

const topicCardStyles = StyleSheet.create({
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 8,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingTop: 10,
    paddingBottom: 6,
    gap: 8,
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: 5,
    flexShrink: 0,
  },
  topicName: {
    flex: 1,
    fontSize: 13,
    fontWeight: '700',
  },
  badge: {
    borderRadius: 10,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  badgeText: {
    fontSize: 11,
    fontWeight: '700',
  },
  rankRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    paddingHorizontal: 10,
    paddingBottom: 10,
    gap: 6,
  },
  rankBtn: {
    borderRadius: 6,
    borderWidth: 1.5,
    paddingHorizontal: 10,
    paddingVertical: 5,
    ...Platform.select({ web: { cursor: 'pointer' as any } }),
  },
  rankBtnText: {
    fontSize: 12,
    fontWeight: '700',
  },
  scoreSection: {
    paddingHorizontal: 12,
    paddingBottom: 10,
    overflow: 'hidden',
  },
  matchScoreBlock: {
    paddingVertical: 4,
    paddingHorizontal: 4,
    marginBottom: 4,
    maxWidth: '100%',
    flexShrink: 1,
  },
  matchScoreLabel: {
    fontSize: 11,
    fontWeight: '700',
    marginBottom: 2,
  },
});

// ─── Root component ───────────────────────────────────────────────────────────

export default function TopicHighlights() {
  const { filename, text, topics, results } = data;
  const [activeMatch, setActiveMatch] = useState<ActiveMatch>(null);
  const initialWidth = Platform.OS === 'web' ? Math.floor((window as any).innerWidth / 2) : 280;
  const [sidebarWidth, setSidebarWidth] = useState(initialWidth);
  const isDragging = useRef(false);
  const sidebarWidthRef = useRef(initialWidth);

  const segments = useMemo(
    () => buildSegments(text, topics, results, activeMatch),
    [text, topics, results, activeMatch],
  );

  // Auto-scroll right panel to highlighted text
  useEffect(() => {
    if (activeMatch === null) return;
    const timer = setTimeout(() => {
      if (Platform.OS === 'web') {
        (document as any).getElementById('active-highlight')
          ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }, 60);
    return () => clearTimeout(timer);
  }, [activeMatch]);

  // Resizable divider
  function startResize(e: any) {
    if (Platform.OS !== 'web') return;
    e.preventDefault();
    isDragging.current = true;
    const startX: number = e.clientX;
    const startWidth: number = sidebarWidthRef.current;
    // Prevent text selection and pointer interference during drag
    if (Platform.OS === 'web') {
      (document.body.style as any).userSelect = 'none';
      (document.body.style as any).pointerEvents = 'none';
    }
    const onMouseMove = (me: MouseEvent) => {
      if (!isDragging.current) return;
      const delta = me.clientX - startX;
      const newWidth = Math.max(120, Math.min(startWidth + delta, (window as any).innerWidth - 120));
      sidebarWidthRef.current = newWidth;
      setSidebarWidth(newWidth);
    };
    const onMouseUp = () => {
      isDragging.current = false;
      if (Platform.OS === 'web') {
        (document.body.style as any).userSelect = '';
        (document.body.style as any).pointerEvents = '';
      }
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('mouseup', onMouseUp);
    };
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('mouseup', onMouseUp);
  }

  function handleMatchSelect(topic: string, rank: number) {
    setActiveMatch((prev) =>
      prev && prev.topic === topic && prev.rank === rank ? null : { topic, rank }
    );
  }

  return (
    <View style={styles.root}>
      <StatusBar style="light" />

      {/* ── Top bar ── */}
      <View style={styles.topBar}>
        <Text style={styles.topTitle}>Topic Retrieval Viewer</Text>
        <Text style={styles.topSub}>
          {filename} · {topics.length} topics
          {activeMatch
            ? `  ·  ${activeMatch.topic} #${activeMatch.rank}`
            : '  ·  click a rank to highlight'}
        </Text>
      </View>

      {/* ── Body: sidebar + document ── */}
      <View style={styles.body}>
        {/* ── Left: topic panel ── */}
        <View style={[styles.sidebarClip, { width: sidebarWidth }]}>
          <ScrollView
            style={styles.sidebar}
            contentContainerStyle={styles.sidebarContent}
          >
            <Text style={styles.panelHeading}>TOPICS</Text>
            {topics.map((topic, i) => (
              <TopicCard
                key={topic}
                topic={topic}
                topicIndex={i}
                matches={results[topic] ?? []}
                activeRank={activeMatch?.topic === topic ? activeMatch.rank : null}
                onMatchSelect={(rank) => handleMatchSelect(topic, rank)}
              />
            ))}
          </ScrollView>
        </View>

        {/* Resizable divider */}
        <View
          style={styles.divider}
          {...(Platform.OS === 'web' ? { onMouseDown: startResize } : {})}
        >
          <View style={styles.dividerHandle} />
        </View>

        {/* ── Right: document with highlights ── */}
        <ScrollView style={styles.docScroll} contentContainerStyle={styles.docScrollContent}>
          {/* Legend strip */}
          <ScrollView
            horizontal
            showsHorizontalScrollIndicator={false}
            style={styles.legendStrip}
            contentContainerStyle={styles.legendStripContent}
          >
            {topics.map((topic, i) => {
              const c = topicColor(i);
              const active = activeMatch === null || activeMatch.topic === topic;
              return (
                <View
                  key={topic}
                  style={[
                    styles.legendChip,
                    { backgroundColor: c.bg, borderColor: c.border },
                    !active && styles.legendChipDim,
                  ]}
                >
                  <View style={[styles.legendDot, { backgroundColor: c.border }]} />
                  <Text style={[styles.legendChipText, { color: c.text }]}>{topic}</Text>
                </View>
              );
            })}
          </ScrollView>

          {/* Paper */}
          <View style={styles.paper}>
            <View style={styles.pdfHeaderRow}>
              <View style={styles.pdfLine} />
              <Text style={styles.pdfFilename} numberOfLines={1}>{filename}</Text>
              <View style={styles.pdfLine} />
            </View>

            <Text style={styles.bodyText}>
              {segments.map((seg, i) => {
                if (seg.topicIndex === null) {
                  return (
                    <Text key={i} style={styles.bodyText}>
                      {seg.text}
                    </Text>
                  );
                }
                const c = topicColor(seg.topicIndex);
                return (
                  <Text
                    key={i}
                    nativeID={activeMatch !== null ? 'active-highlight' : undefined}
                    style={[
                      styles.bodyText,
                      styles.highlight,
                      { backgroundColor: c.bg },
                    ]}
                  >
                    {seg.text}
                  </Text>
                );
              })}
            </Text>
          </View>
        </ScrollView>
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

  // Top bar
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

  // Body split
  body: {
    flex: 1,
    flexDirection: 'row',
  },

  // Sidebar outer clip (enforces dragged width)
  sidebarClip: {
    overflow: 'hidden',
    flexShrink: 0,
    flexGrow: 0,
  },
  // Sidebar scroll area
  sidebar: {
    flex: 1,
    backgroundColor: '#F1F3F4',
  },
  sidebarContent: {
    padding: 12,
    paddingBottom: 40,
  },
  panelHeading: {
    fontSize: 10,
    fontWeight: '700',
    color: '#888',
    letterSpacing: 1,
    marginBottom: 8,
  },

  // Resizable divider
  divider: {
    width: 8,
    backgroundColor: '#E0E3E8',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    ...Platform.select({ web: { cursor: 'col-resize' as any } }),
  },
  dividerHandle: {
    width: 3,
    height: 40,
    borderRadius: 2,
    backgroundColor: '#B0B7C3',
  },

  // Document scroll area
  docScroll: {
    flex: 1,
  },
  docScrollContent: {
    paddingVertical: 20,
    paddingHorizontal: 16,
    alignItems: 'center',
  },

  // Legend strip above doc
  legendStrip: {
    width: '100%',
    maxHeight: 40,
    marginBottom: 12,
  },
  legendStripContent: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 4,
  },
  legendChip: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 14,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderWidth: 1,
    gap: 5,
  },
  legendChipDim: {
    opacity: 0.35,
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
  },
  legendChipText: {
    fontSize: 11,
    fontWeight: '600',
  },

  // Paper (white document card)
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
    lineHeight: 26,
    color: '#1A1A1A',
    fontFamily: Platform.select({
      ios: 'Georgia',
      android: 'serif',
      default: 'Georgia, "Times New Roman", serif',
    }),
  },

  // Highlighted span
  highlight: {
    borderRadius: 2,
    paddingHorizontal: 1,
  },
});
