#!/bin/bash

storescp -su +B +xa --timenames -od $PACS_EGRESS_DIR -xcs "python $RECEIVER_PY_PATH -d #p" -tos $PACS_INGRESS_TIMEOUT $INGRESS_PORT