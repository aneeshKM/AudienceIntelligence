# 01 — Replace the Groq HTTP adapter with the official SDK

**What to build:** Article classification uses one reusable synchronous Groq SDK client instead of the hand-written HTTP integration. The adapter retains strict structured output, returns the provider's raw message content for independent validation and auditing, and keeps retries under application control. Tests can inject an offline fake client, while the existing live Groq check remains opt-in.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] The project declares Groq SDK version `>=1.5,<2`, and the generator constructs one reusable client using the supplied or environment API key, a 60-second timeout, no SDK retries, and the SDK's standard endpoint.
- [ ] The generator accepts an optional injected SDK client so unit tests make no network requests and do not patch SDK internals.
- [ ] Each non-streaming completion uses the configured model (defaulting to `openai/gpt-oss-120b`), medium reasoning effort, a 2048 completion-token limit, and no explicit temperature or top-p value.
- [ ] Each completion forwards the supplied schema as strict `article_judgment` JSON Schema response formatting and sends the classification prompt as the user message.
- [ ] The generator returns the first choice's message content unchanged so existing parsing, validation, audited attempts, and application-level retry behavior continue to work.
- [ ] Focused unit coverage verifies client configuration, request parameters, schema forwarding, client reuse, and unchanged raw-content return behavior using an injected fake SDK client.
- [ ] The focused classification tests and complete offline test suite pass; the existing live Groq integration test remains skipped unless explicitly enabled or invoked directly.
- [ ] The untracked manual streaming experiment remains unchanged.
