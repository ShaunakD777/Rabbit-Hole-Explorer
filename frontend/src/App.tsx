import { useAppStore } from './store/appStore'
import LandingPage from './pages/LandingPage'
import GraphWorkspace from './pages/GraphWorkspace'

export default function App() {
  const topicId = useAppStore((s) => s.topicId)
  return topicId ? <GraphWorkspace /> : <LandingPage />
}
