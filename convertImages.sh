#!/usr/bin/env/ bash
find sample_Signature/genuine/ -name "*.png" -exec convert {} -resize 256x256! -colorspace Gray new/{} \;
