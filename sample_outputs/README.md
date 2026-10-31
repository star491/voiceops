# Sample outputs

Produced by `python run.py --input test_audio --output outputs` (Gnani STT -> agent -> Gnani TTS).
The customer clips were synthesised with Gnani TTS (voice Pranav) from the practice set, so no real voices were recorded.

Per clip:
- `<name>.customer.wav`: the input audio
- `<name>.reply.wav`: the agent's spoken reply (Gnani TTS, voice Kaveri)
- `<name>.reply.txt`: the reply text
- `<name>.json`: the transcript, reply, **section IDs used and their text**, retrieved sections, and tool calls (verification values masked)

`results_all_22_clips.csv` covers every clip in the run. `PC002_turn1..4` is a single conversation:
greeting, order ID, verification, then a damaged TV worth more than 10,000 rupees, which is escalated with a ticket.

| Clip | Heard (Gnani STT) | Agent reply | Sections / data used | Tools |
|---|---|---|---|---|
| `PQ0003` | how fast are refunds to the zentromart wallet | Refunds to your ZentroMart Wallet are instant, usually within 2 hours. | REF-2 | – |
| `PQ0004` | can i schedule a delivery at midnight | Sorry, I don't have information about that. I can connect you to a colleague who can help. | – | – |
| `PQ0008` | credit card vaala refund seventeenth october twenty twenty six ko shuru hua | Your refund should reach your credit card between 24 October 2026 and 27 October 2026. | REF-2, KB-00.3 | date_calc |
| `PQ0010` | is the collection free if the urban thread slim fit jeans came torn | The reverse‑pickup fee is waived for damaged items, so you won’t be charged for collecting the jeans. | products.csv:FSH-0029 | – |
| `PC002_turn1` | hello mujhe order ke baare mein help chaahiye | Namaste, main ZentroMart ka AI assistant hoon. Kripya apna order ID batayenge? | STY-1 | – |
| `PC002_turn2` | order number hai eight nine seven eight two eight three seven | Kripya apne registered phone ka aakhri 4 ank ya delivery pincode batayenge? | IDV-2 | lookup_order |
| `PC002_turn3` | last four digits four thousand five hundred thirty three hai | Ek ticket banaya gaya hai aur Priority Resolution team aapse jald sampark karegi. कृपया आगे के अपडेट के लिए अपना फ़ोन चालू रखें। | ESC-1, ESC-2, orders.csv:89782837 | lookup_order, create_ticket |
| `PC002_turn4` | man is entro vista forty three inch four k smart tv damage report kiya tha abhi replacement approve kar do naa | Damaged TV ke liye aapka report 48 ghante ke andar kiya gaya tha, isliye aapko free replacement ya poora refund mil sakta hai. Humne ticket banaya hai; Priority… | DWM-1, DWM-2, ESC-1 | – |
