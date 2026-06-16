import { useCallback, useState } from "react";

export interface UnsavedChangesController {
  isDirty: boolean;
  markDirty: () => void;
  markClean: () => void;
}

export function useUnsavedChanges(initialDirty: boolean = false): UnsavedChangesController {
  const [isDirty, setIsDirty] = useState<boolean>(initialDirty);

  const markDirty = useCallback(() => {
    setIsDirty(true);
  }, []);

  const markClean = useCallback(() => {
    setIsDirty(false);
  }, []);

  return { isDirty, markDirty, markClean };
}

