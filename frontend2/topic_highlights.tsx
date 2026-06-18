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
  start: number; // character offset in full text
};

type UserAnnotation = { start: number; end: number; topicIndex: number | null };

type SelectionPopupState = { x: number; y: number; start: number; end: number } | null;

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
  userAnnotations: UserAnnotation[] = [],
): Segment[] {
  // Build base intervals
  let baseIntervals: { start: number; end: number; topicIndex: number; rank: number }[] = [];

  if (activeMatch !== null) {
    const matches = results[activeMatch.topic] ?? [];
    const match = matches.find((m) => m.rank === activeMatch.rank);
    if (match) {
      const topicIndex = topics.indexOf(activeMatch.topic);
      baseIntervals = [{ start: match.start, end: match.end, topicIndex, rank: match.rank }];
    }
  } else {
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
    baseIntervals = Array.from(spanMap.entries())
      .map(([startStr, v]) => ({ start: parseInt(startStr, 10), end: v.end, topicIndex: v.topicIndex, rank: v.rank }));
  }

  // User annotations have highest priority — clip base intervals around them so there are no overlaps
  const userIntervals = userAnnotations.map((a) => ({ start: a.start, end: a.end, topicIndex: a.topicIndex, rank: -1 }));

  const clippedBase: typeof baseIntervals = [];
  for (const base of baseIntervals) {
    let remaining: { start: number; end: number }[] = [{ start: base.start, end: base.end }];
    for (const ua of userIntervals) {
      const next: { start: number; end: number }[] = [];
      for (const r of remaining) {
        if (ua.end <= r.start || ua.start >= r.end) {
          next.push(r); // no overlap
        } else {
          if (r.start < ua.start) next.push({ start: r.start, end: ua.start });
          if (r.end > ua.end) next.push({ start: ua.end, end: r.end });
        }
      }
      remaining = next;
    }
    for (const r of remaining) {
      clippedBase.push({ start: r.start, end: r.end, topicIndex: base.topicIndex, rank: base.rank });
    }
  }

  const allIntervals = [...userIntervals, ...clippedBase].sort((a, b) => a.start - b.start);

  const segments: Segment[] = [];
  let cursor = 0;
  for (const span of allIntervals) {
    if (span.start < cursor) continue;
    if (span.start > cursor) segments.push({ text: fullText.slice(cursor, span.start), topicIndex: null, matchRank: 0, start: cursor });
    segments.push({ text: fullText.slice(span.start, span.end), topicIndex: span.topicIndex, matchRank: span.rank, start: span.start });
    cursor = span.end;
  }
  if (cursor < fullText.length) segments.push({ text: fullText.slice(cursor), topicIndex: null, matchRank: 0, start: cursor });
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
      </View>

      {/* Score bars for all matches — make each block clickable */}
      <View style={topicCardStyles.scoreSection}>
        {matches.map((m) => {
          const isSelected = activeRank === m.rank;
          return (
            <TouchableOpacity
              key={m.chunk_index}
              onPress={() => onMatchSelect(m.rank)}
              activeOpacity={0.8}
              style={[
                topicCardStyles.matchScoreBlock,
                isSelected && { backgroundColor: color.bg, borderRadius: 4 },
              ]}
            >
              <Text style={[topicCardStyles.matchScoreLabel, { color: color.text }]}>#{m.rank}</Text>
              <ScoreBar label="rrf"  value={m.rrf_score}       max={0.02} color={color.border} />
              <ScoreBar label="sem"  value={m.semantic_score}  max={1}    color={color.border} />
              <ScoreBar label="bm25" value={m.bm25_score}      max={20}   color={color.border} />
            </TouchableOpacity>
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

// ─── LLM prompt builder ───────────────────────────────────────────────────────

function buildPrompt(
  docText: string,
  topics: string[],
  results: Record<string, TopicMatch[]>,
  userAnnotations: UserAnnotation[],
): string {
  const changeLines = userAnnotations.map((ann) => {
    const excerpt = docText.slice(ann.start, ann.end).trim().slice(0, 120) +
      (ann.end - ann.start > 120 ? '…' : '');
    let origTopic: string | null = null;
    let origMatch: TopicMatch | null = null;
    let bestRank = Infinity;
    topics.forEach((topic) => {
      (results[topic] ?? []).forEach((m) => {
        if (m.start < ann.end && m.end > ann.start && m.rank < bestRank) {
          bestRank = m.rank;
          origTopic = topic;
          origMatch = m;
        }
      });
    });
    const newTopic = ann.topicIndex !== null ? topics[ann.topicIndex] : null;
    const m = origMatch as TopicMatch | null;
    const scores = m
      ? `rrf=${m.rrf_score.toFixed(4)}, semantic=${m.semantic_score.toFixed(4)}, bm25=${m.bm25_score.toFixed(4)}`
      : 'no original match';

    if (origTopic === null && newTopic !== null) {
      return `- MISSED MATCH: text "${excerpt}" should belong to topic "${newTopic}" but the system did not assign any topic (no match found).`;
    } else if (origTopic !== null && newTopic === null) {
      return `- FALSE POSITIVE: text "${excerpt}" was wrongly assigned to topic "${origTopic}" (scores: ${scores}) — it has no relevant topic.`;
    } else if (origTopic !== null && newTopic !== null) {
      return `- WRONG TOPIC: text "${excerpt}" was assigned to "${origTopic}" (scores: ${scores}) but the correct topic is "${newTopic}".`;
    }
    return '';
  }).filter(Boolean);

  console.log('[DiagnosisPrompt] Human corrections sent to LLM:');
  changeLines.forEach((line) => console.log(line));

  const changes = changeLines.join('\n');

  return `You are an expert in RAG (Retrieval-Augmented Generation) systems and information retrieval.

## Task background
A document has been processed by a hybrid topic-matching pipeline that uses Reciprocal Rank Fusion (RRF) combining:
- Dense semantic embeddings (cosine similarity)
- BM25 lexical scoring

The pipeline matches text spans in the document to a set of topics. A human reviewer has corrected the automatic assignments.

## Topics in the system
${topics.map((t, i) => `${i + 1}. ${t}`).join('\n')}

## Human corrections (Before → After)
${changes}

## Your task
For each correction write one numbered entry (2-3 sentences max): failure mode, why the pipeline erred, targeted fix.

Close with a 1-2 sentence summary of the dominant pattern and the single most impactful improvement.

Strict rules: plain prose only — no markdown headers, no bold, no bullet symbols, no horizontal rules. Maximum 160 words total. Cut anything obvious.`;
}

// ─── Simple markdown → plain-text renderer ───────────────────────────────────

function stripInline(text: string): string {
  return text
    .replace(/\*\*\*(.+?)\*\*\*/g, '$1')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/\*(.+?)\*/g, '$1')
    .replace(/`(.+?)`/g, '$1')
    .replace(/\[(.+?)\]\(.+?\)/g, '$1');
}

type MdBlock =
  | { type: 'heading'; level: number; text: string }
  | { type: 'bullet'; text: string }
  | { type: 'numbered'; n: number; text: string }
  | { type: 'paragraph'; text: string };

function parseMarkdown(raw: string): MdBlock[] {
  const blocks: MdBlock[] = [];
  for (const line of raw.split('\n')) {
    const t = line.trim();
    if (!t || t === '---' || t === '***' || t === '___') continue;
    const m3 = t.match(/^###\s+(.*)/);
    if (m3) { blocks.push({ type: 'heading', level: 3, text: stripInline(m3[1]) }); continue; }
    const m2 = t.match(/^##\s+(.*)/);
    if (m2) { blocks.push({ type: 'heading', level: 2, text: stripInline(m2[1]) }); continue; }
    const m1 = t.match(/^#\s+(.*)/);
    if (m1) { blocks.push({ type: 'heading', level: 1, text: stripInline(m1[1]) }); continue; }
    const bull = t.match(/^[-*+]\s+(.*)/);
    if (bull) { blocks.push({ type: 'bullet', text: stripInline(bull[1]) }); continue; }
    const num = t.match(/^(\d+)\.\s+(.*)/);
    if (num) { blocks.push({ type: 'numbered', n: parseInt(num[1], 10), text: stripInline(num[2]) }); continue; }
    blocks.push({ type: 'paragraph', text: stripInline(t) });
  }
  return blocks;
}

function MarkdownView({ raw, showCursor }: { raw: string; showCursor: boolean }) {
  const blocks = parseMarkdown(raw);
  return (
    <View>
      {blocks.map((block, i) => {
        const isLast = i === blocks.length - 1;
        const suffix = isLast && showCursor ? '▋' : '';
        if (block.type === 'heading') {
          const sz = block.level === 1 ? 14 : block.level === 2 ? 13 : 12;
          return (
            <Text key={i} style={[mdStyles.heading, { fontSize: sz }]}>
              {block.text}{suffix}
            </Text>
          );
        }
        if (block.type === 'bullet') {
          return (
            <View key={i} style={mdStyles.listRow}>
              <Text style={mdStyles.listMarker}>•</Text>
              <Text style={mdStyles.listText}>{block.text}{suffix}</Text>
            </View>
          );
        }
        if (block.type === 'numbered') {
          return (
            <View key={i} style={mdStyles.listRow}>
              <Text style={mdStyles.listMarker}>{block.n}.</Text>
              <Text style={mdStyles.listText}>{block.text}{suffix}</Text>
            </View>
          );
        }
        return (
          <Text key={i} style={mdStyles.paragraph}>{block.text}{suffix}</Text>
        );
      })}
    </View>
  );
}

const mdStyles = StyleSheet.create({
  heading: {
    color: '#B0BAFF',
    fontWeight: '700',
    marginTop: 10,
    marginBottom: 3,
    lineHeight: 20,
  },
  listRow: {
    flexDirection: 'row',
    marginBottom: 5,
    paddingLeft: 2,
  },
  listMarker: {
    color: '#6868A8',
    fontSize: 12,
    lineHeight: 20,
    marginRight: 6,
    minWidth: 16,
  },
  listText: {
    color: '#D0D0F0',
    fontSize: 12,
    lineHeight: 20,
    flex: 1,
  },
  paragraph: {
    color: '#D0D0F0',
    fontSize: 12,
    lineHeight: 20,
    marginBottom: 7,
  },
});

// ─── Diagnosis chatbot component ──────────────────────────────────────────────

const GROQ_API_KEY = (process.env as any).EXPO_PUBLIC_GROQ_API_KEY ?? '';
const GROQ_MODEL = 'openai/gpt-oss-20b';
const GROQ_MAX_COMPLETION_TOKENS = 1024;
// Typing speed: characters advanced per tick
const CHARS_PER_TICK = 5;
const TICK_MS = 28;

function contentToText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .map((part: any) => {
        if (typeof part === 'string') return part;
        if (typeof part?.text === 'string') return part.text;
        if (typeof part?.content === 'string') return part.content;
        return '';
      })
      .join('');
  }
  return '';
}

function DiagnosisChatbot({
  docText,
  topics,
  results,
  userAnnotations,
}: {
  docText: string;
  topics: string[];
  results: Record<string, TopicMatch[]>;
  userAnnotations: UserAnnotation[];
}) {
  const [displayed, setDisplayed] = useState('');
  const [isTyping, setIsTyping] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cursorOn, setCursorOn] = useState(true);
  const [panelWidth, setPanelWidth] = useState(480);
  const scrollRef = useRef<any>(null);
  const abortRef = useRef<AbortController | null>(null);
  // Buffer for received text; display advances slowly via timer
  const receivedRef = useRef('');
  const displayCountRef = useRef(0);
  const streamDoneRef = useRef(false);
  const panelWidthRef = useRef(480);
  const isDraggingChatRef = useRef(false);

  // Stream from Groq on mount — accumulate into receivedRef
  useEffect(() => {
    const prompt = buildPrompt(docText, topics, results, userAnnotations);
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    const requestBody = {
      model: GROQ_MODEL,
      stream: true,
      max_completion_tokens: GROQ_MAX_COMPLETION_TOKENS,
      reasoning_effort: 'low',
      reasoning_format: 'hidden',
      messages: [{ role: 'user', content: prompt }],
    };

    // Abort if no response starts within 15 s
    const timeoutId = setTimeout(() => ctrl.abort(), 15000);

    (async () => {
      try {
        const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
          method: 'POST',
          signal: ctrl.signal,
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${GROQ_API_KEY}`,
          },
          body: JSON.stringify(requestBody),
        });

        if (!res.ok) {
          const body = await res.text();
          throw new Error(`Groq API error ${res.status}: ${body}`);
        }

        const reader = (res as any).body.getReader();
        const decoder = new (globalThis as any).TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';
          for (const line of lines) {
            const trimmed = line.trim();
            if (!trimmed || trimmed === 'data: [DONE]') continue;
            if (!trimmed.startsWith('data: ')) continue;
            try {
              const json = JSON.parse(trimmed.slice(6));
              const token = contentToText(json.choices?.[0]?.delta?.content);
              if (token) receivedRef.current += token;
            } catch {
              // malformed chunk — skip
            }
          }
        }
        clearTimeout(timeoutId);
        streamDoneRef.current = true;
        // Guard: if the stream completed but returned nothing, show an error
        if (!receivedRef.current) {
          const fallbackRes = await fetch('https://api.groq.com/openai/v1/chat/completions', {
            method: 'POST',
            signal: ctrl.signal,
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${GROQ_API_KEY}`,
            },
            body: JSON.stringify({ ...requestBody, stream: false }),
          });

          if (!fallbackRes.ok) {
            const body = await fallbackRes.text();
            throw new Error(`Groq API error ${fallbackRes.status}: ${body}`);
          }

          const fallbackJson = await fallbackRes.json();
          const fallbackText = contentToText(fallbackJson.choices?.[0]?.message?.content);
          if (fallbackText) {
            receivedRef.current = fallbackText;
          } else {
            const finishReason = fallbackJson.choices?.[0]?.finish_reason ?? 'unknown';
            setIsTyping(false);
            setError(`No response from model. Finish reason: ${finishReason}.`);
          }
        }
      } catch (err: any) {
        clearTimeout(timeoutId);
        if (err?.name === 'AbortError') {
          setIsTyping(false);
          setError('Request timed out. Check your connection and try again.');
          return;
        }
        setIsTyping(false);
        setError(err?.message ?? 'Unknown error');
      }
    })();

    return () => { ctrl.abort(); clearTimeout(timeoutId); };
  // Run once on mount only
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Slow-typing timer: advance displayed text by CHARS_PER_TICK every TICK_MS ms
  useEffect(() => {
    const timer = setInterval(() => {
      const target = receivedRef.current.length;
      if (displayCountRef.current < target) {
        displayCountRef.current = Math.min(displayCountRef.current + CHARS_PER_TICK, target);
        const slice = receivedRef.current.slice(0, displayCountRef.current);
        setDisplayed(slice);
        setTimeout(() => scrollRef.current?.scrollToEnd?.({ animated: false }), 0);
      } else if (streamDoneRef.current) {
        setIsTyping(false);
      }
    }, TICK_MS);
    return () => clearInterval(timer);
  }, []);

  // Blinking cursor
  useEffect(() => {
    const blink = setInterval(() => setCursorOn((v) => !v), 530);
    return () => clearInterval(blink);
  }, []);

  // Resize handle (on the left edge of the panel — drag left to widen)
  function startPanelResize(e: any) {
    if (Platform.OS !== 'web') return;
    e.preventDefault();
    isDraggingChatRef.current = true;
    const startX: number = e.clientX;
    const startW: number = panelWidthRef.current;
    const doc = (globalThis as any).document;
    if (doc) {
      doc.body.style.userSelect = 'none';
      doc.body.style.pointerEvents = 'none';
    }
    const onMove = (me: any) => {
      if (!isDraggingChatRef.current) return;
      const delta = startX - me.clientX; // drag left = increase width
      const newW = Math.max(300, Math.min(startW + delta, 900));
      panelWidthRef.current = newW;
      setPanelWidth(newW);
    };
    const onUp = () => {
      isDraggingChatRef.current = false;
      if (doc) {
        doc.body.style.userSelect = '';
        doc.body.style.pointerEvents = '';
        doc.removeEventListener('mousemove', onMove);
        doc.removeEventListener('mouseup', onUp);
      }
    };
    if (doc) {
      doc.addEventListener('mousemove', onMove);
      doc.addEventListener('mouseup', onUp);
    }
  }

  return (
    <View style={[chatStyles.panel, { width: panelWidth }]}>
      {/* Left-edge resize handle */}
      <View
        style={chatStyles.resizeHandle}
        {...(Platform.OS === 'web' ? { onMouseDown: startPanelResize } : {})}
      >
        <View style={chatStyles.resizeHandleBar} />
      </View>

      {/* Inner content */}
      <View style={chatStyles.inner}>
        {/* Header */}
        <View style={chatStyles.header}>
          <View style={chatStyles.headerIcon}>
            <Text style={chatStyles.headerIconText}>✦</Text>
          </View>
          <Text style={chatStyles.headerTitle}>AI Diagnosis</Text>
          {isTyping && !error && (
            <View style={chatStyles.typingBadge}>
              <Text style={chatStyles.typingBadgeText}>typing…</Text>
            </View>
          )}
        </View>

        {/* Message area */}
        <ScrollView
          ref={scrollRef}
          style={chatStyles.scroll}
          contentContainerStyle={chatStyles.scrollContent}
          showsVerticalScrollIndicator={true}
        >
          {error ? (
            <View style={chatStyles.errorBubble}>
              <Text style={chatStyles.errorText}>⚠ {error}</Text>
            </View>
          ) : displayed === '' ? (
            <View style={chatStyles.thinkingRow}>
              <Text style={chatStyles.thinkingText}>Analyzing topic changes</Text>
              <Text style={chatStyles.thinkingDots}>   . . .</Text>
            </View>
          ) : (
            <View style={chatStyles.bubble}>
              <MarkdownView raw={displayed} showCursor={isTyping && cursorOn} />
            </View>
          )}
        </ScrollView>
      </View>
    </View>
  );
}

// ─── Root component ───────────────────────────────────────────────────────────

export default function TopicHighlights() {
  const { filename, text, topics, results } = data;
  const [activeMatch, setActiveMatch] = useState<ActiveMatch>(null);
  const [selectedTopicIndexes, setSelectedTopicIndexes] = useState<number[]>([]);
  const [userAnnotations, setUserAnnotations] = useState<UserAnnotation[]>([]);
  const [selectionPopup, setSelectionPopup] = useState<SelectionPopupState>(null);
  const [submitted, setSubmitted] = useState(false);
  // Maps nativeID "seg-{i}" → character start offset in full text
  const segStartsRef = useRef<Record<string, number>>({});
  const initialWidth = Platform.OS === 'web' && typeof (globalThis as any).window !== 'undefined'
    ? Math.floor(((globalThis as any).window as any).innerWidth / 2)
    : 280;
  const [sidebarWidth, setSidebarWidth] = useState(initialWidth);
  const isDragging = useRef(false);
  const sidebarWidthRef = useRef(initialWidth);

  const segments = useMemo(
    () => buildSegments(text, topics, results, activeMatch, userAnnotations),
    [text, topics, results, activeMatch, userAnnotations],
  );

  const originalSegments = useMemo(
    () => buildSegments(text, topics, results, null, []),
    [text, topics, results],
  );

  const submittedSegments = useMemo(
    () => buildSegments(text, topics, results, null, userAnnotations),
    [text, topics, results, userAnnotations],
  );

  // Text selection → topic picker popup (web only)
  useEffect(() => {
    if (Platform.OS !== 'web') return;
    const doc: any = (globalThis as any).document;
    if (!doc) return;

    const handleMouseUp = (e: any) => {
      // Ignore clicks inside the popup itself
      if (e?.target?.closest?.('[data-selection-popup]')) return;

      // Use rAF so the browser has fully committed the selection by the time we read it
      ;(globalThis as any).requestAnimationFrame(() => {
        const docObj: any = (globalThis as any).document;
        const sel: any = (globalThis as any).window?.getSelection?.();
        if (!sel || sel.isCollapsed || !sel.toString().trim()) {
          setSelectionPopup(null);
          return;
        }

        try {
          const range: any = sel.getRangeAt(0);
          const rect: any = range.getBoundingClientRect();
          if (!rect || (rect.width === 0 && rect.height === 0)) return;

          // Find the nearest ancestor element with id="seg-{n}"
          const getSegEl = (node: any): any => {
            let el: any = node.nodeType === 3 ? node.parentElement : node;
            while (el) {
              if ((el.id ?? '').startsWith('seg-') && el.id in segStartsRef.current) return el;
              el = el.parentElement;
            }
            return null;
          };

          // Compute absolute char offset: create a Range from [segEl,0] to
          // [container,domOffset] and call toString() — the browser handles all
          // nested-span / Element-vs-TextNode ambiguity internally.
          // Do NOT strip \n: the source text may contain real newlines, and all
          // seg-N spans are inline so the browser never injects synthetic ones.
          const getDocOffset = (container: any, domOffset: number): number | null => {
            const segEl = getSegEl(container);
            if (!segEl) return null;
            const segStart: number = segStartsRef.current[segEl.id];
            try {
              const r = docObj.createRange();
              r.setStart(segEl, 0);
              r.setEnd(container, domOffset);
              return segStart + r.toString().length;
            } catch {
              return null;
            }
          };

          const docStart = getDocOffset(range.startContainer, range.startOffset);
          const docEnd   = getDocOffset(range.endContainer,   range.endOffset);
          if (docStart === null || docEnd === null || docEnd <= docStart) return;

          setSelectionPopup({
            x: rect.left + rect.width / 2,
            y: rect.bottom + 8,
            start: docStart,
            end: docEnd,
          });
        } catch {
          // ignore
        }
      });
    };

    const handleKeyDown = (e: any) => {
      if (e?.key === 'Escape') setSelectionPopup(null);
    };

    doc.addEventListener('mouseup', handleMouseUp);
    doc.addEventListener('keydown', handleKeyDown);
    return () => {
      doc.removeEventListener('mouseup', handleMouseUp);
      doc.removeEventListener('keydown', handleKeyDown);
    };
  }, []);

  function applyAnnotation(topicIndex: number | null) {
    if (!selectionPopup) return;
    setUserAnnotations((prev) => [
      ...prev.filter((a) => !(a.start < selectionPopup.end && a.end > selectionPopup.start)), // remove overlapping
      { start: selectionPopup.start, end: selectionPopup.end, topicIndex },
    ]);
    setSelectionPopup(null);
    // Clear browser selection
    (globalThis as any).window?.getSelection?.()?.removeAllRanges?.();
  }

  // Auto-scroll right panel to highlighted text
  useEffect(() => {
    if (activeMatch === null) return;
    const timer = setTimeout(() => {
      if (Platform.OS === 'web' && typeof (globalThis as any).document !== 'undefined') {
        // Find the seg-{i} id for the active match segment
        const activeSegIdx = segments.findIndex(
          (seg) => seg.topicIndex !== null && seg.matchRank === activeMatch.rank
        );
        if (activeSegIdx !== -1) {
          ((globalThis as any).document as any)
            .getElementById(`seg-${activeSegIdx}`)
            ?.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }
    }, 60);
    return () => clearTimeout(timer);
  }, [activeMatch, segments]);

  // Resizable divider
  function startResize(e: any) {
    if (Platform.OS !== 'web') return;
    e.preventDefault();
    isDragging.current = true;
    const startX: number = (e as any).clientX;
    const startWidth: number = sidebarWidthRef.current;
    // Prevent text selection and pointer interference during drag
    if (typeof (globalThis as any).document !== 'undefined') {
      ((globalThis as any).document.body.style as any).userSelect = 'none';
      ((globalThis as any).document.body.style as any).pointerEvents = 'none';
    }
    const onMouseMove = (me: any) => {
      if (!isDragging.current) return;
      const delta = me.clientX - startX;
      const maxW = typeof (globalThis as any).window !== 'undefined' ? ((globalThis as any).window as any).innerWidth - 120 : 1200;
      const newWidth = Math.max(120, Math.min(startWidth + delta, maxW));
      sidebarWidthRef.current = newWidth;
      setSidebarWidth(newWidth);
    };
    const onMouseUp = () => {
      isDragging.current = false;
      if (typeof (globalThis as any).document !== 'undefined') {
        ((globalThis as any).document.body.style as any).userSelect = '';
        ((globalThis as any).document.body.style as any).pointerEvents = '';
        ((globalThis as any).document as any).removeEventListener('mousemove', onMouseMove);
        ((globalThis as any).document as any).removeEventListener('mouseup', onMouseUp);
      }
    };
    if (typeof (globalThis as any).document !== 'undefined') {
      ((globalThis as any).document as any).addEventListener('mousemove', onMouseMove);
      ((globalThis as any).document as any).addEventListener('mouseup', onMouseUp);
    }
  }

  function handleMatchSelect(topic: string, rank: number) {
    setActiveMatch((prev) =>
      prev && prev.topic === topic && prev.rank === rank ? null : { topic, rank }
    );
  }

  function handleTopicToggle(topicIndex: number) {
    setSelectedTopicIndexes((prev) =>
      prev.includes(topicIndex)
        ? prev.filter((i) => i !== topicIndex)
        : [...prev, topicIndex]
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

      {/* ── Body: sidebar + document  OR  diff view ── */}
      {submitted ? (
        /* ── Diff view: Before (left) vs After (right) ── */
        <View style={styles.diffContainer}>
          {/* toolbar */}
          <View style={styles.diffToolbar}>
            <TouchableOpacity
              onPress={() => setSubmitted(false)}
              style={styles.backBtn}
              activeOpacity={0.8}
            >
              <Text style={styles.backBtnText}>← Back to Edit</Text>
            </TouchableOpacity>
            <Text style={styles.diffToolbarTitle}>Comparison View</Text>
            <View style={{ width: 110 }} />
          </View>

          <View style={styles.diffBody}>
            {/* Before column */}
            <View style={styles.diffColWrapper}>
              <View style={[styles.diffColHeader, styles.diffColHeaderBefore]}>
                <Text style={styles.diffColHeading}>Before</Text>
              </View>
              <ScrollView style={styles.diffScroll} contentContainerStyle={styles.diffScrollContent}>
                <View style={styles.paper}>
                  <View style={styles.pdfHeaderRow}>
                    <View style={styles.pdfLine} />
                    <Text style={styles.pdfFilename} numberOfLines={1}>{filename}</Text>
                    <View style={styles.pdfLine} />
                  </View>
                  <Text style={styles.bodyText}>
                    {originalSegments.map((seg, i) => {
                      if (seg.topicIndex === null) {
                        return <Text key={i} style={styles.bodyText}>{seg.text}</Text>;
                      }
                      const c = topicColor(seg.topicIndex);
                      return (
                        <Text key={i} style={[styles.bodyText, styles.highlight, { backgroundColor: c.bg }]}>
                          {seg.text}
                        </Text>
                      );
                    })}
                  </Text>
                </View>
              </ScrollView>
            </View>

            {/* Column divider */}
            <View style={styles.diffColDivider} />

            {/* After column */}
            <View style={styles.diffColWrapper}>
              <View style={[styles.diffColHeader, styles.diffColHeaderAfter]}>
                <Text style={styles.diffColHeading}>After</Text>
              </View>
              <ScrollView style={styles.diffScroll} contentContainerStyle={styles.diffScrollContent}>
                <View style={styles.paper}>
                  <View style={styles.pdfHeaderRow}>
                    <View style={styles.pdfLine} />
                    <Text style={styles.pdfFilename} numberOfLines={1}>{filename}</Text>
                    <View style={styles.pdfLine} />
                  </View>
                  <Text style={styles.bodyText}>
                    {submittedSegments.map((seg, i) => {
                      if (seg.topicIndex === null) {
                        return <Text key={i} style={styles.bodyText}>{seg.text}</Text>;
                      }
                      const c = topicColor(seg.topicIndex);
                      return (
                        <Text key={i} style={[styles.bodyText, styles.highlight, { backgroundColor: c.bg }]}>
                          {seg.text}
                        </Text>
                      );
                    })}
                  </Text>
                </View>
              </ScrollView>
            </View>

            {/* Column divider */}
            <View style={styles.diffColDivider} />

            {/* Chatbot panel */}
            <DiagnosisChatbot
              docText={text}
              topics={topics}
              results={results}
              userAnnotations={userAnnotations}
            />
          </View>
        </View>
      ) : (
      /* ── Body: sidebar + document ── */
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
        <View style={styles.docWrapper}>
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
              const selected = selectedTopicIndexes.length === 0 || selectedTopicIndexes.includes(i);
              return (
                <TouchableOpacity
                  key={topic}
                  onPress={() => handleTopicToggle(i)}
                  activeOpacity={0.8}
                  style={[
                    styles.legendChip,
                    { backgroundColor: c.bg, borderColor: c.border },
                    !selected && styles.legendChipDim,
                  ]}
                >
                  <View style={[styles.legendDot, { backgroundColor: c.border }]} />
                  <Text style={[styles.legendChipText, { color: c.text }]}>{topic}</Text>
                </TouchableOpacity>
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
                const segId = `seg-${i}`;
                // Keep ref map up to date (safe to mutate during render for web-only lookup)
                segStartsRef.current[segId] = seg.start;
                if (seg.topicIndex === null) {
                  return (
                    <Text
                      key={i}
                      nativeID={segId}
                      style={styles.bodyText}
                    >
                      {seg.text}
                    </Text>
                  );
                }
                if (
                  selectedTopicIndexes.length > 0 &&
                  !selectedTopicIndexes.includes(seg.topicIndex)
                ) {
                  return (
                    <Text
                      key={i}
                      nativeID={segId}
                      style={styles.bodyText}
                    >
                      {seg.text}
                    </Text>
                  );
                }
                const c = topicColor(seg.topicIndex);
                return (
                  <Text
                    key={i}
                    nativeID={segId}
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
        {/* ── Submit button ── */}
        {userAnnotations.length > 0 && (
          <TouchableOpacity
            style={styles.submitBtn}
            onPress={() => setSubmitted(true)}
            activeOpacity={0.85}
          >
            <Text style={styles.submitBtnText}>Submit ›</Text>
          </TouchableOpacity>
        )}
        </View>
      </View>
      )}
      {/* ── Selection → topic picker popup (web only) ── */}
      {selectionPopup !== null && Platform.OS === 'web' && (
        <View
          data-selection-popup
          style={[
            popupStyles.container,
            {
              left: selectionPopup.x - 160,
              top: selectionPopup.y,
            },
          ]}
        >
          <Text style={popupStyles.heading}>Assign topic colour</Text>
          <View style={popupStyles.grid}>
            <TouchableOpacity
              onPress={() => applyAnnotation(null)}
              activeOpacity={0.8}
              style={[popupStyles.topicBtn, popupStyles.decolorBtn]}
            >
              <View style={[popupStyles.dot, popupStyles.decolorDot]} />
              <Text style={[popupStyles.topicBtnText, popupStyles.decolorText]}>Remove topic</Text>
            </TouchableOpacity>
            {topics.map((topic, i) => {
              const c = topicColor(i);
              return (
                <TouchableOpacity
                  key={topic}
                  onPress={() => applyAnnotation(i)}
                  activeOpacity={0.8}
                  style={[popupStyles.topicBtn, { backgroundColor: c.bg, borderColor: c.border }]}
                >
                  <View style={[popupStyles.dot, { backgroundColor: c.border }]} />
                  <Text style={[popupStyles.topicBtnText, { color: c.text }]} numberOfLines={1}>
                    {topic}
                  </Text>
                </TouchableOpacity>
              );
            })}
          </View>
          <TouchableOpacity onPress={() => setSelectionPopup(null)} style={popupStyles.cancelBtn}>
            <Text style={popupStyles.cancelText}>Cancel</Text>
          </TouchableOpacity>
        </View>
      )}
    </View>
  );
}

// ─── Selection popup styles ───────────────────────────────────────────────────

const popupStyles = StyleSheet.create({
  container: {
    position: 'fixed' as any,
    width: 320,
    backgroundColor: '#FFFFFF',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#D0D5DD',
    padding: 14,
    zIndex: 1000,
    ...Platform.select({
      web: {
        // @ts-ignore
        boxShadow: '0 8px 24px rgba(0,0,0,0.18)',
      },
    }),
  },
  heading: {
    fontSize: 11,
    fontWeight: '700',
    color: '#888',
    letterSpacing: 0.8,
    marginBottom: 10,
    textTransform: 'uppercase' as any,
  },
  grid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
    marginBottom: 10,
  },
  topicBtn: {
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 1.5,
    borderRadius: 20,
    paddingHorizontal: 10,
    paddingVertical: 5,
    gap: 5,
    maxWidth: '100%',
    ...Platform.select({ web: { cursor: 'pointer' as any } }),
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    flexShrink: 0,
  },
  topicBtnText: {
    fontSize: 12,
    fontWeight: '600',
    flexShrink: 1,
  },
  cancelBtn: {
    alignSelf: 'flex-end',
    paddingHorizontal: 10,
    paddingVertical: 4,
    ...Platform.select({ web: { cursor: 'pointer' as any } }),
  },
  cancelText: {
    fontSize: 12,
    color: '#888',
  },
  decolorBtn: {
    backgroundColor: '#F5F5F5',
    borderColor: '#BDBDBD',
  },
  decolorDot: {
    backgroundColor: '#9E9E9E',
  },
  decolorText: {
    color: '#616161',
  },
});

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

  // Wrapper for right panel (positions submit button)
  docWrapper: {
    flex: 1,
    position: 'relative' as any,
  },

  // Submit button — fixed bottom-right of right panel
  submitBtn: {
    position: 'absolute' as any,
    bottom: 24,
    right: 24,
    backgroundColor: '#1A1A2E',
    borderRadius: 8,
    paddingHorizontal: 22,
    paddingVertical: 11,
    ...Platform.select({
      web: {
        // @ts-ignore
        boxShadow: '0 4px 14px rgba(0,0,0,0.25)',
        cursor: 'pointer' as any,
      },
    }),
  },
  submitBtnText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '700',
    letterSpacing: 0.3,
  },

  // Diff / comparison view
  diffContainer: {
    flex: 1,
    flexDirection: 'column',
  },
  diffToolbar: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#F1F3F4',
    borderBottomWidth: 1,
    borderBottomColor: '#D0D5DD',
    paddingHorizontal: 16,
    paddingVertical: 10,
  },
  diffToolbarTitle: {
    fontSize: 13,
    fontWeight: '700',
    color: '#444',
    letterSpacing: 0.3,
  },
  backBtn: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    backgroundColor: '#1A1A2E',
    borderRadius: 6,
    ...Platform.select({ web: { cursor: 'pointer' as any } }),
  },
  backBtnText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '700',
  },
  diffBody: {
    flex: 1,
    flexDirection: 'row',
  },
  diffColWrapper: {
    flex: 1,
    flexDirection: 'column',
  },
  diffColHeader: {
    paddingHorizontal: 16,
    paddingVertical: 7,
  },
  diffColHeaderBefore: {
    backgroundColor: '#FFF3E0',
    borderBottomWidth: 2,
    borderBottomColor: '#FFA726',
  },
  diffColHeaderAfter: {
    backgroundColor: '#E8F5E9',
    borderBottomWidth: 2,
    borderBottomColor: '#66BB6A',
  },
  diffColHeading: {
    fontSize: 11,
    fontWeight: '700',
    letterSpacing: 1,
    color: '#555',
    textTransform: 'uppercase' as any,
  },
  diffColDivider: {
    width: 2,
    backgroundColor: '#D0D5DD',
    flexShrink: 0,
  },
  diffScroll: {
    flex: 1,
  },
  diffScrollContent: {
    paddingVertical: 20,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
});

// ─── Chatbot styles ───────────────────────────────────────────────────────────

const chatStyles = StyleSheet.create({
  panel: {
    flexShrink: 0,
    backgroundColor: '#0F0F1A',
    flexDirection: 'row',
  },
  resizeHandle: {
    width: 8,
    backgroundColor: '#1A1A30',
    alignItems: 'center',
    justifyContent: 'center',
    flexShrink: 0,
    ...Platform.select({ web: { cursor: 'col-resize' as any } }),
  },
  resizeHandleBar: {
    width: 3,
    height: 40,
    borderRadius: 2,
    backgroundColor: '#3E3E6E',
  },
  inner: {
    flex: 1,
    flexDirection: 'column',
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#1A1A2E',
    borderBottomWidth: 1,
    borderBottomColor: '#2E2E4E',
  },
  headerIcon: {
    width: 22,
    height: 22,
    borderRadius: 11,
    backgroundColor: '#4F46E5',
    alignItems: 'center',
    justifyContent: 'center',
  },
  headerIconText: {
    color: '#FFFFFF',
    fontSize: 11,
    fontWeight: '700',
  },
  headerTitle: {
    color: '#E0E0FF',
    fontSize: 13,
    fontWeight: '700',
    flex: 1,
  },
  typingBadge: {
    backgroundColor: '#2E2E4E',
    borderRadius: 8,
    paddingHorizontal: 7,
    paddingVertical: 2,
  },
  typingBadgeText: {
    color: '#9090C0',
    fontSize: 10,
    fontStyle: 'italic',
  },
  scroll: {
    flex: 1,
  },
  scrollContent: {
    padding: 14,
    paddingBottom: 24,
  },
  thinkingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 12,
  },
  thinkingText: {
    color: '#9090C0',
    fontSize: 13,
    fontStyle: 'italic',
  },
  thinkingDots: {
    color: '#4F46E5',
    fontSize: 16,
    letterSpacing: 2,
  },
  bubble: {
    backgroundColor: '#1C1C30',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#2E2E4E',
    padding: 12,
  },
  messageText: {
    color: '#D0D0F0',
    fontSize: 12,
    lineHeight: 20,
  },
  errorBubble: {
    backgroundColor: '#2A0A0A',
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#5C1A1A',
    padding: 12,
  },
  errorText: {
    color: '#FF8080',
    fontSize: 12,
    lineHeight: 18,
  },
});
