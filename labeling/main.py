#!/usr/bin/env python3

import sys
from labeling_app import LabelingApp

def main() -> int:
    # Create and run the labeling application
    app = LabelingApp()
    return app.run()

if __name__ == "__main__":
    sys.exit(main())
