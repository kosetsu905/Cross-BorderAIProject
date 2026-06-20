# Support Inbox Live E2E Scenarios

The machine-readable source is `references/scenarios.json`. This file explains the suites and the expected behavior in human terms.

Use `RUN_ID` as a unique value such as `20260620-153000`. Every subject and body includes a marker in this form:

```text
CB-SUPPORT-E2E-{scenario}-{run_id}
```

Recipient: the Gmail inbox configured by `GMAIL_SENDER_EMAIL` in the project `.env`.

## Suites

- `balanced`: the default 12-message regression suite.
- `smoke`: the original 4-message core suite.
- `--scenarios`: a comma-separated custom list, for example `pre_sales_catalog_headset,order_tracking_found`.

## Balanced Scenarios

### pre_sales_catalog_headset

Wireless Bluetooth headset catalog facts, feature coverage, and discount approval.

Expected: `pre_sales`; draft includes `$6.50`, `40PCS`, `52x38x51cm`, `9 kg`; no JSON/code fence; discount must require sales review or approval.

### pre_sales_catalog_m90

M90 PRO wireless earphones catalog price and bulk discount request.

Expected: `pre_sales`; draft includes `$2.39`; discount must require sales approval; no invented reduced price.

### pre_sales_feature_unknown

Wireless Bluetooth headset exact battery, Bluetooth version, microphone, latency, sound quality, and color specs.

Expected: `pre_sales`; draft should say exact specs are not listed/available instead of inventing features such as Bluetooth 5.3, long battery life, or ANC.

### pre_sales_unknown_product

Quantum Solar Drone X9000 price, carton, stock, and variants.

Expected: `pre_sales`; draft must not invent catalog price, stock, SKU, or facts from headset/earphone catalog items; it should ask for model/SKU/exact item confirmation or route for review.

### order_tracking_found

Correct local tracking number `C88943021`.

Expected: `order_fulfillment`; `tracking_record_found == true`; draft includes `C88943021`, `120399587991`, and delivery facts; it must not frame the answer as not found.

### order_tracking_not_found

Wrong tracking number `C99943021`.

Expected: `order_fulfillment`; not-found tracking status; draft asks the customer to verify the tracking/order details; conversation must not leak `C88943021`, `120399587991`, or the correct local tracking facts.

### order_no_identifier

Customer asks where their package is without any order, reference, email, or tracking id.

Expected: `order_fulfillment`; draft asks for tracking number, order ID, reference number, or purchase email; it must not guess status or use local tracking facts.

### order_marker_pollution

Customer gives `C99943021` while the body also contains the live test marker.

Expected: `order_fulfillment`; system treats only `C99943021` as the business tracking number; draft/payload must not treat the marker date/run id as tracking data.

### post_sales_worn_bra

Worn bra change-of-mind refund request.

Expected: `post_sales_support`; draft explains hygiene/intimate item risk; no refund approval, prepaid return label, return tracking number, or immediate refund promise.

### post_sales_defective_item

Recently received defective item, asking for refund/replacement and RMA review.

Expected: `post_sales_support`; draft asks for proof/photos or explains RMA review; it must not invent return logistics unless structured `logistics_output` exists, and must never leak `labels.example.local`.

### post_sales_return_window_expired

Unopened item received 45 days ago, customer asks for refund and label.

Expected: `post_sales_support`; draft explains return-window risk or review path; no direct approval, prepaid label, or refund promise.
It must not say "eligible for a return", "pleased to inform you", or "return approved" when the 45-day window contradicts the policy.

### post_sales_zh_escalation

Chinese complaint: item is broken, customer wants refund and manager/supervisor contact.

Expected: `post_sales_support`; Chinese or Chinese-adapted reply; escalation/handoff signal should be present.

## Global Acceptance Criteria

- Every selected marker conversation is found after Gmail sync.
- `draft_response` is plain customer-facing text, not raw JSON or fenced JSON.
- Drafts must not echo the live test marker.
- `detected_intent` matches the scenario.
- Required catalog/tracking facts appear when expected.
- Forbidden local facts, fabricated prices, return labels, and `labels.example.local` do not appear.
- Report records `conversation_status`, `requires_approval`, `outbound_message_count`, `outbound_channel_message_ids`, and `auto_dispatch_observed`.
- By default, each scenario should have no more than one outbound auto-dispatch message. More than one outbound message is a failure because it usually means Gmail self-sent replies were re-synced as new inbound work.
