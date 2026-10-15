# CatAI inference service (Vercel + Backblaze B2)

This folder is a separately deployable Vercel Python project. Set the Vercel
project **Root Directory** to \`inference\` in the CatAI repository. It uses
NumPy for inference rather than shipping the training-only PyTorch dependency
inside the serverless function.

## Runtime environment variables

Configure these in the inference Vercel project (Production and Preview as
appropriate):

- \`CATAI_API_SECRET\`: long random bearer secret shared only with Banana Data Center.
- \`B2_ENDPOINT_URL\`: Backblaze S3-compatible endpoint. The runtime accepts
  either the bare hostname \`s3.us-east-005.backblazeb2.com\` or the full URL
  \`https://s3.us-east-005.backblazeb2.com\`; do not include bucket/object paths.
- \`B2_REGION\`: region matching the B2 endpoint, e.g. \`us-east-005\`.
- \`B2_KEY_ID\`: restricted Backblaze application key ID with read access to the model bucket.
- \`B2_APPLICATION_KEY\`: matching restricted application key.
- \`B2_BUCKET_NAME\`: private bucket holding CatAI checkpoints.
- \`CATAI_MODEL_KEY\` (optional): active manifest key, default
  \`ai/models/current.json\`.
- \`CATAI_MODEL_IDLE_SECONDS\` (optional): local model cache idle timeout,
  default 60 seconds (minimum 15).

Do not expose any of these values to browser code. The bucket should stay
private and the runtime key should have read-only access.

## Publish secrets in GitHub Actions

Add the following Actions secrets to the CatAI repository before dispatching
the full training workflow:

- \`B2_KEY_ID\`
- \`B2_APPLICATION_KEY\`
- \`B2_BUCKET_NAME\`
- \`B2_ENDPOINT_URL\`
- \`B2_REGION\`

The workflow exports both the original training checkpoint
(\`ai/models/catai-sft.pt\`, optimizer state included for future fine-tuning)
and a compact inference bundle. It uploads the inference bundle under a
commit-specific key, verifies byte count and SHA-256, and only then updates
\`ai/models/current.json\` to point to the new revision.

## Request contract

POST to \`/api/generate\` with a private bearer token:

\`\`\`json
{
  "messages": [
    {"role": "system", "content": "You are CatAI, an honest and helpful assistant."},
    {"role": "user", "content": "Hello"}
  ],
  "max_new_tokens": 160,
  "temperature": 0.8,
  "top_k": 40
}
\`\`\`

The response contains \`text\`, \`model_revision\`, \`tokens_in\`,
\`tokens_out\`, and \`context_limit\`. The endpoint does not save conversations;
Banana Data Center owns the authenticated conversation history in Upstash.

## Local cache behavior

The active manifest is read only when a model needs to be loaded. The selected
bundle downloads into the function's temporary directory, is SHA-256 verified,
and is loaded into process memory. After 60 seconds of inactivity, a best-effort
timer clears the in-memory arrays and unlinks the local model file. A request
also checks for an expired cache before loading. Vercel may freeze or recycle a
function without running cleanup hooks; temporary storage is never authoritative,
and a future cold start downloads the active model again.

## Important scope

The current SFT artifact's context is only 768 character-level tokens. This
deployment gives CatAI on-demand model hosting and durable conversations; it
does not magically provide an 8-billion-token transformer context window.
That requires a different long-context model architecture and training plan.
