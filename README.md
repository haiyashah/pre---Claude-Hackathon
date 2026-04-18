# Faculty — A Personal Mentor for Creative Flourishing

> Built at the CMU x Anthropic Claude Builder Club Hackathon, April 2026
> Theme: Creative Flourishing

---

## What Is This?

Most people never find out what they could have made, built, or become — not because they lack talent, but because they never had the right person asking them the right questions at the right time.

Faculty is an AI-powered mentorship system for teenagers (ages 14 to 17) that interviews users the way a great therapist or mentor would: not "what do you like to do?" but "when did you feel most alive?", "what did you make as a kid?", "what do you secretly envy in others?"

The output is not a playlist or a personality label. It is a living map of your unlived creative self, rendered as an interactive knowledge graph — with multiple specialist mentor voices ready to speak to each node on the map. A human counselor and a parent remain in the loop throughout, ensuring the system stays grounded in who the child actually is.

The name Faculty is intentional. Not one AI assistant, but a coordinated team of specialist mentors — like having a full faculty dedicated to a single student.

---

## The Problem This Solves

Access to genuine mentorship is deeply unequal. Kids who get into IIT, CMU, and top programs often credit not just hard work but a peer group and a mentor who showed them how to think differently — how to question the assumptions everyone else accepted.

Most kids never get that. Faculty is built to change that.

There is also a deeper problem the tool addresses: what Anthropic CEO Dario Amodei described as "herding behavior masquerading as maturity and sophistication" — the way apparent consensus can shape what people believe is possible for themselves. Faculty helps users find their own map, not someone else's.

---

## How Claude Is Used (For Judges)

This project uses Claude deeply and in multiple layers. A full Google Doc with screenshots, API call breakdowns, and prompting notes is linked here: [INSERT GOOGLE DOC LINK]

### 1. The Interview Engine — `index.html` + `/api/claude`

The opening interview is a carefully structured sequence of questions designed to surface:
- Moments of peak aliveness and flow
- Childhood creative instincts before socialization
- Envy as a signal of unlived potential
- Discomfort and avoidance as diagnostic signals
- The gap between current life and imagined life

Each answer is sent to Claude via the `/api/claude` proxy endpoint using the CMU AI Gateway. Claude synthesizes all answers into a structured JSON graph: a root node, thematic branches, leaf nodes representing unlived directions, and concrete action nodes.

Example API call (from `index.html`):
```javascript
fetch('/api/claude', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: analysisPrompt }],
    max_tokens: 2000
  })
})
```

The system prompt used for graph generation instructs Claude to respond in strict JSON — no preamble, no markdown fences — with a schema that includes `root`, `branches`, `leaves`, and `actions`. This structured output is then rendered as an interactive D3.js force-directed graph.

### 2. The Faculty Voices — `/api/faculty`

Every node on the graph can be clicked. When a node is clicked, the server calls Claude using one of six specialist "voices," selected based on the node's domain:

| Voice | Domain | Behavior |
|---|---|---|
| The Artist Voice | Creative | Speaks in images and possibility. Opens doors. |
| The Strategist Voice | Career | Pragmatic, direct, no flattery, concrete next steps. |
| The Socratic Voice | Purpose | Asks the question beneath the question. Ends with two questions. |
| The Mirror Voice | Identity | Shows users what they already know but have not said aloud. |
| The Connector Voice | Relationship | Speaks about loneliness, belonging, the need for the right people. |
| The Pragmatist Voice | Action | Breaks the unlived thing into the smallest possible first move. |

Each voice has a fully engineered system prompt that shapes tone, length, and method. The prompt includes the user's full answer history as context, plus the specific node label and source quote that triggered the click.

Example voice prompt structure (simplified):
```
You are a Socratic questioner. Ask the question under the question. End with 2 questions.

Context:
Q1: [user answer]
Q2: [user answer]
...

Exploring: "The company not started"
Their words: "I kept waiting to feel ready."

Respond as The Socratic Voice. Be specific. No em dashes.
```

Faculty responses are cached in the session and persisted to disk, so the map grows richer over time.

### 3. The Continuing Conversation — `chat.html`

After the map is generated, users can continue talking to the Faculty via a full chat interface. The chat loads the user's session (including their archetype name, all answers, and all previous faculty responses) and uses this as context for every new message.

The chat supports RAG over indexed documents — counselors or parents can upload guidelines, reading lists, or frameworks as PDFs, and the system will draw on them when responding. This ensures the mentor speaks in alignment with the values of the humans who know this child.

Claude is called with the full conversation history on every message:
```javascript
messages: [...conversationHistory, { role: 'user', content: userInput }]
```

### 4. Session Persistence — `/api/session/save` and `/api/session/load`

Every session is saved as a JSON file on the server. Sessions include: the archetype name, all interview answers, the full graph structure, all faculty voice responses, and all chat messages. This enables the longitudinal mentorship model — the system gets to know the user over time.

### 5. The Dev/Counselor Interface — `dev.html`

The `/dev` route is the counselor's interface. Counselors and parents can:
- Upload PDF guidelines that shape how the mentor responds (indexed via a Milvus vector store)
- Clear and rebuild the index
- Export indexed chunks for review
- Monitor RAG status

This is the human-in-the-loop layer. The agent amplifies the humans who know this child. It does not replace them.

---

## Architecture Overview

```
Browser (index.html)
    |
    | Interview answers
    v
Flask Server (server.py)
    |
    |-- /api/claude ---------> CMU AI Gateway --> Claude Sonnet 4
    |                          (graph generation, chat)
    |
    |-- /api/faculty --------> CMU AI Gateway --> Claude Sonnet 4
    |                          (specialist voice responses)
    |
    |-- /api/rag/query ------> Milvus Vector Store
    |                          (counselor-uploaded PDFs)
    |
    |-- /api/session/* ------> Local JSON files
                               (session persistence)
```

The frontend (index.html) renders the knowledge graph using D3.js force simulation. Nodes are colored by domain. Clicking any node opens a side panel that calls the Faculty endpoint and streams back a specialist voice response.

---

## Running Locally

**Requirements:** Python 3.9+, pip

```bash
# Install dependencies
pip install flask requests

# Optional: for RAG features
pip install pymilvus sentence-transformers pypdf2

# Set your API key (or use the CMU gateway key in server.py)
export CMU_API_KEY=your-key-here

# Run
python server.py
```

Then open:
- `http://localhost:8501` — The interview and knowledge graph
- `http://localhost:8501/chat` — The continuing conversation
- `http://localhost:8501/dev` — The counselor/parent upload interface

---

## The Human-in-the-Loop Design

Faculty is not a product that runs autonomously on children. The architecture is intentional:

**Counselors** upload guidelines that shape mentor behavior. If a counselor believes a particular student needs encouragement toward structure, or challenge toward risk, they configure that through the RAG layer. The agent executes their philosophy, not the model's default.

**Parents** receive thematic summaries of their child's sessions — not transcripts, but progress signals. The mentor shares themes like "exploring questions about creative identity" rather than surfacing raw conversation content. This preserves the trust needed for honest engagement while keeping parents informed.

**Students** interact with the system knowing it has this layer. This is not surveillance. It is a supported environment, analogous to a school counselor who checks in with parents at the right level of abstraction.

This design directly addresses the core risk of any "think for yourself" tool: without human grounding, the tool's own biases become the new consensus. With humans in the loop, the system amplifies chosen mentors — not an anonymous model.

---

## Age Rationale

Faculty is designed for ages 14 to 17. This is the window where:
- Identity questions become real and urgent, not destabilizing
- The gap between peer pressure and individual instinct is at its peak
- College applications and career questions create concrete decision pressure
- Parents are still legally and emotionally present in the child's life
- Intervention has the highest leverage

Under 14 raises different developmental and regulatory questions that are outside the scope of this demo. The 14 to 17 window is also autobiographically motivated: this is when good mentorship has the most transformative effect, and when most students do not have access to it.

---

## What "Creative Flourishing" Means Here

The hackathon theme asks: as AI handles routine work, what happens to human purpose?

Faculty's answer is that purpose is not found through productivity tools. It is found through the right questions, asked at the right time, by someone who takes you seriously.

The system does not tell users what to create. It reflects back what they already know about themselves, and then offers the smallest possible next move. The Faculty voices disagree with each other by design — the Socratic voice asks questions while the Pragmatist says "here is what you do on Monday." That productive tension is the point.

A student who completes the interview and explores two or three nodes on their graph will have spent more time thinking seriously about their own creative potential than most people do in years.

---

## Team

Built at CMU x Anthropic Claude Builder Club Hackathon, April 18, 2026.

---

## Links

- GitHub Repo: [INSERT LINK]
- Devpost: [INSERT LINK]
- Claude API Usage Documentation (Google Doc): [INSERT LINK]
