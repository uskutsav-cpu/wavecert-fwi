# Historical test reuse: descriptive parent comparison

Prior inspected roles; 16 ID slices / 8 parents and 4 OOD slices / 1 parent. All five controls, all 24 blocks.

| Fixed model | ID: equal-parent mean cosine | OOD: one-parent mean cosine |
| --- | ---: | ---: |
| Legacy FNO: forward | -0.0445 | -0.0732 |
| Legacy FNO: JVP | -0.0581 | -0.0809 |
| New FNO: validation selected | 0.3554 | -0.4543 |
| POD64: uncorrected | 0.9998 | 0.9999 |
| POD64 + coarse96: corrected | 0.9998 | 0.9999 |

Parent means average each parent's own slices first. No IID intervals, ID/OOD pooling, new-population inference, or nominal conformal claim. One OOD parent cannot establish broad OOD reliability. Ordinary floating point.
