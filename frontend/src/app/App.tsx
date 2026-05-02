import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";
import { SessionForecastViewProvider } from "../features/sessions/context/SessionForecastViewContext";

export default function App() {
  return (
    <BrowserRouter>
      <SessionForecastViewProvider>
        <AppRoutes />
      </SessionForecastViewProvider>
    </BrowserRouter>
  );
}
