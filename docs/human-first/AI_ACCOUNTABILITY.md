# AI Accountability Principles

CatAI believes that AI developers and organizations deploying AI systems must take responsibility for the effects of their products. Releasing an AI feature does not remove the responsibility to protect users, explain limitations, prevent foreseeable damage, and provide ways to address failures.

These principles apply to CatAI itself as well as to external AI providers, agent platforms, and organizations integrating AI into their products.

## 1. Keep the user in control

AI systems should make it clear what they are about to do and what they have done. Actions that can materially change, publish, delete, transfer, or expose a user's data or work should require clear authorization appropriate to the risk.

For AI agents that edit code or manage projects, responsible safeguards include:

- Showing a preview or clear summary of proposed changes.
- Asking before destructive, irreversible, or broadly scoped actions.
- Respecting the user's selected files, directories, settings, and boundaries.
- Avoiding unrelated changes outside the requested task.
- Making it easy to stop an operation.

A user's request for help should not be interpreted as permission to make unlimited changes.

## 2. Protect user work and provide recovery

AI products that can modify user work should make recovery a core feature, not an afterthought. Providers should support appropriate safeguards such as:

- Backups, snapshots, version history, or equivalent recovery points.
- Clear diffs and change histories for agent-made edits.
- Reliable rollback or restore mechanisms.
- Protection against accidental deletion and overwriting.
- Clear instructions for recovering from failed or unwanted actions.

Where a product cannot guarantee recovery, that limitation should be explained before users entrust important work to it.

## 3. Be transparent about behavior and limitations

Providers should explain what their systems can access, what actions they can perform, when they use external services, and what limitations or uncertainty apply. Marketing claims should not imply reliability, autonomy, or safety beyond what has been demonstrated.

Users should be able to distinguish suggestions from actions already performed.

## 4. Keep humans accountable

AI systems should not become a way for organizations to avoid responsibility. A named organization and accountable human process should remain available for consequential outcomes, errors, complaints, and appeals.

Human oversight must be meaningful. People reviewing an AI result need adequate context, authority, and time to challenge it.

## 5. Make AI optional where practical

Providers should avoid making AI mandatory for unrelated or routine product functionality. They should preserve non-AI workflows or human support where practical, and explain when a feature genuinely depends on AI.

Users should not be punished for choosing a non-AI route when one can reasonably be provided.

## 6. Protect privacy and security

Providers should limit access to the data needed for a task, protect sensitive information, use secure defaults, and explain relevant data handling. Agent permissions should be scoped to the user's request and should not silently expand.

Products should provide useful records of important agent actions while handling logs and personal data responsibly.

## 7. Respond responsibly when things go wrong

When an AI system causes or contributes to a failure, its provider should:

1. Make it possible for the user to stop further automated actions.
2. Preserve relevant diagnostic information without unnecessarily exposing private data.
3. Explain what happened, what is known, and what remains uncertain.
4. Help the user recover their work where possible.
5. Provide a clear route for support, review, and escalation.
6. Use incident findings to improve safeguards and communicate material fixes.

Providers should not dismiss user reports simply because the system was expected to be helpful or because the failure was unintended.

## 8. CatAI's own responsibility

CatAI should hold its own products to these standards. It should be honest about limitations, avoid overstating safeguards that are not implemented, and prioritize user control, privacy, and recoverability as its capabilities grow.

If CatAI cannot provide a safeguard described here, it should say so clearly rather than implying that the safeguard already exists.

## 9. Industry engagement

CatAI may share these principles with AI companies and software platforms. This includes raising concerns with developer-agent providers such as Replit about destructive changes, authorization, project backups, and recovery.

Any public account of an incident or company response should distinguish verified facts from personal reports, interpretations, and recommendations. CatAI should not imply that a company has responded, agreed, or changed its product unless that outcome is confirmed.

## Related documents

- [`HUMAN_FIRST.md`](./HUMAN_FIRST.md) — Core human-first philosophy.
- [`USAGE_POLICY.md`](./USAGE_POLICY.md) — Expectations for people and organizations using CatAI.
- [`AI_ADVOCACY.md`](./AI_ADVOCACY.md) — How CatAI may engage with governments and companies.
