#!/usr/bin/env sh
# Fetches the third-party pYIN implementations used by the benchmarks at pinned revisions.
# They are NOT part of fastpitch (different licences: c4dm/pyin is GPL-2.0).
set -e
cd "$(dirname "$0")"
mkdir -p third_party
cd third_party
fetch() {  # name url commit
  if [ ! -d "$1" ]; then git clone --quiet "$2" "$1"; fi
  git -C "$1" checkout --quiet "$3"
  echo "$1 @ $3"
}
fetch pyin            https://github.com/c4dm/pyin.git                    1d68cac9c5ecb1d81054d25d90158e5838dd78c2
fetch vamp-plugin-sdk https://github.com/vamp-plugins/vamp-plugin-sdk.git 44c2487763eb248a933e9eff9169cfadee375009
fetch sw_libpyin      https://github.com/Sleepwalking/libpyin.git         b38135390b335c3e8cea6ef35cf5093789b36dac
fetch libgvps         https://github.com/Sleepwalking/libgvps.git         2f1b4106d72f8f8138dc447bf0123820c0772cbd
