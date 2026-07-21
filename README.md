# Course Summary AI Service

An AI-powered service that generates structured, topic-based summaries of legal course materials.

## Features

- Fetches course material from the Lawticha API
- Generates summaries organized by topic and subtopic
- Configurable word limits (100-2000 words)
- Uses free, open-source AI models
- Runs on CPU for free hosting on Render

## API Endpoints

### POST /api/v1/summary/generate

Generate a summary for a course material.

**Request Body:**
```json
{
    "slug": "formation-of-a-contract-offer-acceptance",
    "max_words": 500
}