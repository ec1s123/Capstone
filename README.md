# Premier Predict

Premier Predict is a React and Vite web application for exploring Premier League match predictions, model confidence, betting-market comparisons, table projections, and club-level performance trends.

## Deployment

Deployment URL: Add the final hosted application URL here before submission.

The project includes React Router fallback configuration for static deployment:

- `vercel.json` rewrites all routes to `index.html` for Vercel.
- `public/_redirects` rewrites all routes to `index.html` for Netlify.

These files prevent direct page loads such as `/matches`, `/results`, or `/club` from returning a 404 in production.

## Features

- Overview dashboard with season controls, model highlights, table signals, and quick navigation.
- Upcoming matches page with fixture grouping, model probabilities, confidence levels, and match context.
- Results page for completed fixtures with scores, prediction accuracy, model picks, and detailed match panels.
- Talking points page that turns model edges, market disagreements, and result surprises into presentation-ready insights.
- Tables page comparing current standings against predicted standings.
- Club page for team-specific results, form, expected points, confidence, and model accuracy.
- Methodology page explaining the model pipeline, feature design, leakage controls, backtest metrics, and season coverage.
- Model output page for inspecting fixture-level probabilities and model-vs-market differences.

## Model Explanation

The prediction workflow is built around pre-match information. Historical Premier League fixture CSV files are loaded, cleaned, normalized, and converted into structured match records. Team names are standardized so that results, standings, logos, and generated fixtures all refer to clubs consistently.

The model uses a softmax-style three-outcome classifier for home win, draw, and away win probabilities. Inputs include team identity, date context, and bookmaker odds converted into probability-oriented features. Post-match statistics such as goals, shots, cards, corners, fouls, and half-time scores are intentionally excluded from training to avoid target leakage.

The application reads generated prediction outputs from CSV and JSON data files in `src/data`. Frontend helper modules then turn those outputs into:

- Fixture-level model picks and confidence values.
- Actual and predicted league tables.
- Model-vs-market edge calculations.
- Team form and expected-points summaries.
- Upcoming-match projections based on current team profiles.

## Project Structure

```text
.
  Dockerfile                      Production Docker image for the frontend
  docker-compose.yml              Runs the production container on localhost:8080
  nginx.conf                      Nginx config with React Router fallback
  package.json                    React/Vite scripts and frontend dependencies
  requirements.txt                Python dependencies for regenerating ML outputs
  Prem-2026-2003/                 Historical Premier League CSV source files
src/
  App.jsx                         Main routing and app-level state
  MLMODEL.py                      Model training and prediction generation script
  components/                     Reusable layout, table, match, and UI components
  constants/                      Shared navigation and table-column settings
  data/                           Fixture, prediction, metrics, and upcoming-match data
  lib/                            Formatting, standings, prediction, team, and match utilities
  pages/                          Route-level pages for each user workflow
  styles.css                      Global styling and responsive layout utilities
```

## Local Setup

Run commands from the project root:

```bash
cd /Users/adam/Documents/Capstone
```

Install dependencies:

```bash
npm ci
```

Run the development server:

```bash
npm run dev
```

Build for production:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

## Docker Setup

Run commands from the project root:

```bash
cd /Users/adam/Documents/Capstone
```

Build and run the production app in Docker:

```bash
docker compose up --build
```

Then open:

```text
http://localhost:8080
```

`docker compose up --build` runs in the foreground, so closing the terminal or pressing `Ctrl+C` stops the container. To keep it running in the background, use:

```bash
docker compose up --build -d
```

Check whether the container is running:

```bash
docker compose ps
```

Stop the background container:

```bash
docker compose down
```

You can also build and run without Compose:

```bash
docker build -t premier-predict .
docker run --rm -p 8080:80 premier-predict
```

The Docker image builds the Vite app with Node.js, then serves the compiled `dist` files through Nginx. React Router routes such as `/matches`, `/results`, and `/club` are handled by the Nginx fallback in `nginx.conf`.

The Python virtual environment is not required to run the Dockerized app. It is only needed if you want to run `src/MLMODEL.py` locally to regenerate prediction and metrics files.

## Deployment Notes

For Vercel, use:

- Framework preset: Vite
- Build command: `npm run build`
- Output directory: `dist`

For Netlify, use:

- Build command: `npm run build`
- Publish directory: `dist`

After deploying, update the Deployment URL section in this README with the live link.
