                 PRETRAINING
                     │
                     ▼
          ┌─────────────────────┐
          │     TinyStories     │
          │         +           │
          │     FineWeb-Edu     │
          └──────────┬──────────┘
                     │
                     ▼
             BASE LANGUAGE MODEL
                     │
                     │
              SFT / CHAT TRAINING
                     │
          ┌──────────┼───────────┐
          ▼          ▼           ▼
      SmolTalk   No Robots     Dolly
          │          │           │
          └──────────┼───────────┘
                     ▼
              CHAT MODEL
                     │
                     ▼
             User ↔ AI assistant