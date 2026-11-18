import type { Citation, EvidenceBundle, EvidenceTraceEntry } from '@/types/api'
import { IconEvidence } from './icons'
import { SectionTitle } from './primitives'

/**
 * What the agent looked up, and what it found.
 *
 * The tool calls are shown, not just their output. A citation list alone asks
 * the reader to trust that the searches were relevant; showing the queries
 * lets them judge that for themselves — and makes it obvious when the agent
 * went looking for the wrong thing.
 *
 * The ordering mirrors the loop: queries first in the order they ran, then the
 * sources they returned. Sources the agent opened in full are marked, because
 * "appeared in a result list" and "was actually read" are different claims.
 */

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

/* The label names the corpus, not the endpoint. "patents" tells a reader what
   question the call was answering; "tavily_search" tells them nothing. */
const TOOL_LABEL: Record<string, string> = {
  search_literature: 'literature',
  search_patents: 'patents',
  search_clinical_news: 'clinical',
  read_sources: 'read',
  crawl_source: 'crawl',
}

const TOOL_TONE: Record<string, string> = {
  search_literature: 'bg-accent-soft text-accent',
  search_patents: 'bg-warn-soft text-warn',
  search_clinical_news: 'bg-info-soft text-info',
  read_sources: 'border border-rule bg-sunken text-ink-2',
  crawl_source: 'border border-rule bg-sunken text-ink-2',
}

function TraceRow({ entry, index }: { entry: EvidenceTraceEntry; index: number }) {
  const failed = Boolean(entry.error)
  const label = TOOL_LABEL[entry.tool] ?? entry.tool
  const detail = entry.query ?? entry.url ?? `${entry.urls?.length ?? 0} source(s)`

  return (
    <li className="flex items-baseline gap-2.5 py-1">
      <span className="numeral w-4 shrink-0 text-right text-micro text-ink-3">{index + 1}</span>
      <span
        className={`badge shrink-0 ${
          failed ? 'border border-rule bg-sunken text-ink-3' : (TOOL_TONE[entry.tool] ?? 'bg-accent-soft text-accent')
        }`}
      >
        {label}
      </span>
      <span className="min-w-0 flex-1 truncate text-2xs text-ink-2" title={detail}>
        {detail}
      </span>
      <span className="numeral shrink-0 text-micro text-ink-3">
        {failed ? entry.error : `${entry.results ?? entry.source_ids?.length ?? 0} new`}
      </span>
    </li>
  )
}

function SourceRow({ citation }: { citation: Citation }) {
  return (
    <li className="group flex gap-2.5 py-2">
      <span
        id={`source-${citation.id}`}
        className="numeral mt-px shrink-0 scroll-mt-24 text-micro font-medium text-accent"
      >
        {citation.id}
      </span>
      <div className="min-w-0 flex-1">
        <a
          href={citation.url}
          target="_blank"
          rel="noopener noreferrer"
          className="block truncate text-2xs text-ink transition-colors hover:text-accent"
          title={citation.title}
        >
          {citation.title}
        </a>
        <div className="mt-0.5 flex items-center gap-1.5">
          <span className="truncate text-micro text-ink-3">{hostOf(citation.url)}</span>
          {citation.corpus && citation.corpus !== 'literature' && (
            <span className="badge border border-rule bg-sunken text-micro text-ink-3">
              {citation.corpus}
            </span>
          )}
          {citation.read_in_full && (
            <span className="badge border border-rule bg-sunken text-micro text-ink-3">
              read in full
            </span>
          )}
        </div>
      </div>
    </li>
  )
}

interface Props {
  evidence: EvidenceBundle | null
  grounding: boolean
}

export default function EvidencePanel({ evidence, grounding }: Props) {
  if (!grounding && !evidence) return null

  const citations = evidence?.citations ?? []
  const trace = evidence?.trace ?? []

  return (
    <section className="surface p-6 sm:p-7">
      <SectionTitle
        icon={<IconEvidence />}
        aside={
          evidence ? (
            <span className="badge border border-rule bg-sunken text-ink-3">
              <span className="numeral">{evidence.api_calls}</span> call
              {evidence.api_calls === 1 ? '' : 's'}
              {evidence.credits > 0 && (
                <>
                  {' · '}
                  <span className="numeral">{evidence.credits}</span> credits
                </>
              )}
              {evidence.cache_hits > 0 && (
                <>
                  {' · '}
                  <span className="numeral">{evidence.cache_hits}</span> cached
                </>
              )}
            </span>
          ) : (
            <span className="badge bg-accent-soft text-accent">
              <span className="h-1.5 w-1.5 animate-pulse-dot rounded-full bg-accent" />
              searching
            </span>
          )
        }
      >
        Prior art &amp; liabilities
      </SectionTitle>

      {grounding && !evidence && (
        <p className="text-2xs leading-relaxed text-ink-3">
          The agent is deciding what to look up across three corpora — published literature, patent
          claims, and clinical news — and reading what it finds.
        </p>
      )}

      {evidence && (
        <>
          {trace.length > 0 && (
            <div className="mb-5">
              <div className="label mb-1.5">Tool calls, in the order the agent made them</div>
              <ul className="divide-y divide-rule/60">
                {trace.map((entry, i) => (
                  <TraceRow key={i} entry={entry} index={i} />
                ))}
              </ul>
            </div>
          )}

          {citations.length > 0 ? (
            <div className="border-t border-rule pt-4">
              <div className="label mb-1">
                Sources · cited as <span className="numeral normal-case">[S1]</span> in the summary
              </div>
              <ul className="divide-y divide-rule/60">
                {citations.map((c) => (
                  <SourceRow key={c.id} citation={c} />
                ))}
              </ul>
            </div>
          ) : (
            <p className="text-2xs leading-relaxed text-ink-3">
              {evidence.note || 'No literature sources were found for this result.'}
            </p>
          )}

          <p className="mt-5 border-t border-rule pt-4 text-micro leading-relaxed text-pretty text-ink-3">
            The pipeline answers whether a molecule should bind and whether it should be safe. These
            two questions it cannot answer, because they are facts about the world rather than
            properties of a structure: is this already claimed, and has this chemotype already
            failed. Both run after the screening is frozen and reach only the written summary —
            they cannot change a prediction, a filter or a ranking. Citation markers that do not
            resolve to a source listed here are stripped before the summary is shown.
          </p>
        </>
      )}
    </section>
  )
}
