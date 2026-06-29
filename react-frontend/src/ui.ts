import { create } from 'zustand'

export type Page = 'home' | 'robot' | 'obsaction' | 'rewards' | 'training' | 'evaluation' | 'settings' | 'logs'

export interface Toast { id: number; title: string; body: string; severity: string }

const ACCENTS = ['teal', 'ocean', 'violet', 'magenta', 'crimson', 'amber', 'lime', 'slate'] as const
export const ACCENT_NAMES = ACCENTS

const lsGet = (k: string, d: string) => { try { return localStorage.getItem(k) || d } catch { return d } }
let tid = 1

interface UiState {
  page: Page
  theme: 'dark' | 'light'
  accent: string
  dockCollapsed: boolean
  toasts: Toast[]
  setPage: (p: Page) => void
  setTheme: (t: 'dark' | 'light') => void
  setAccent: (a: string) => void
  toggleDock: () => void
  pushToast: (t: Omit<Toast, 'id'>) => void
  dismissToast: (id: number) => void
}

function apply(theme: string, accent: string) {
  const el = document.documentElement
  el.setAttribute('data-theme', theme)
  el.setAttribute('data-accent', accent)
}

const theme0 = lsGet('rtg.theme', 'dark') as 'dark' | 'light'
const accent0 = lsGet('rtg.accent', 'teal')
apply(theme0, accent0)

export const useUi = create<UiState>((set, get) => ({
  page: 'home',
  theme: theme0,
  accent: accent0,
  dockCollapsed: window.innerWidth < 1100,
  toasts: [],
  setPage: (page) => set({ page }),
  setTheme: (theme) => { apply(theme, get().accent); try { localStorage.setItem('rtg.theme', theme) } catch {}; set({ theme }) },
  setAccent: (accent) => { apply(get().theme, accent); try { localStorage.setItem('rtg.accent', accent) } catch {}; set({ accent }) },
  toggleDock: () => set((s) => ({ dockCollapsed: !s.dockCollapsed })),
  pushToast: (t) => {
    const id = tid++
    set((s) => ({ toasts: [...s.toasts, { ...t, id }] }))
    setTimeout(() => get().dismissToast(id), 6000)
  },
  dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((x) => x.id !== id) })),
}))
