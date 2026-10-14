# UI redesign

Reading this as a personal desktop workspace for shared environments, with a calm utility language and a restrained green identity. The product website uses the same typography, color, and voice.

## Audit

The previous interface used Inter, violet/cyan accents, mesh effects, a long combined configuration/action/enrollment page, and seven personal-workspace sections. Several controls used inline styles and browser alerts. Core functionality worth preserving included demo inputs, real separation, uploads, enrollment, identity checks, metrics, personal context, and optional response engines.

## Changes

Voice studio, People, Workspaces, and Model settings have distinct navigation. The existing root and profiles-page routes remain, and model settings adds a separate route. Speaker commands come from the backend. Reference-source demo mode is labelled explicitly.

The interface uses Geist, off-white surfaces, forest green accents, an 8px spacing rhythm, 7px controls, and 12px panels. Semantic CSS variables provide corresponding dark colors. Fonts and Phosphor icons are served locally, with their licenses included. Motion is limited to control feedback and honors reduced-motion preferences.

Every form has labels, errors appear as status messages, destructive voice deletion has a native confirmation dialog, and workspace panels keep their active section after edits. Reports escape untrusted content, expose response sources, and can be exported. Each person's chat history stays separate for the browser session.

Ollama and local OpenAI-compatible runtimes support discovery, saved configuration, and explicit connection errors. Legacy model backends remain available. Electron owns a loopback Python process and uses a sandboxed renderer with Node integration disabled.

## Verification

Backend tests cover adapters, loopback restrictions, history/context, unavailable runtimes, and profile creation. Browser checks cover key workflows, content escaping, both themes, and 390px, 800px, and 1440px layouts. Desktop checks exercise startup, navigation, and renderer isolation. A real llama3.2:3b response was obtained from the installed Ollama runtime.

The desktop package requires an installed Python environment and audio dependencies. Smart-home devices remain simulated. Demo speech generation uses online TTS. The showcase repository uses actual app screenshots and explicitly labels its example conversation.
