# Notes

## Current experiment decision

Use only the accepted curated aligned features for the first cross-domain shift run.

## Included aligned features

- protocol
- service
- flow_duration
- source_to_destination_packets
- destination_to_source_packets
- source_to_destination_bytes
- destination_to_source_bytes
- source_mean_bytes_per_packet
- destination_mean_bytes_per_packet

## Deferred for sensitivity analysis

- connection_state (`conn_state` ↔ `state`)

## Questions to check later

- Are IoT-23 `orig_*` features directionally equivalent to UNSW `s*` features?
- Are byte features defined consistently across both datasets?
- Does including the state mapping help or hurt transfer?

## End goal
Build the cross-domain dataset views
IoT-23 and UNSW must be reduced to the same aligned columns.

Train in both directions

IoT-23 -> UNSW-NB15
UNSW-NB15 -> IoT-23
Record the outcomes

macro F1
attack recall
attack F1
confusion matrices
predictions
feature importances
Interpret the results
This is the real thesis payoff:

which direction transfers worse?
which features seem stable?
which mappings are weak?
does cross-domain failure support the domain-shift argument?