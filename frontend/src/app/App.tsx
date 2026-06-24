import { BrowserRouter } from "react-router-dom";
import { AppRoutes } from "./routes";
import { SessionForecastViewProvider } from "../features/sessions/context/SessionForecastViewContext";

const routerBasename = import.meta.env.BASE_URL === "/" ? undefined : import.meta.env.BASE_URL.replace(/\/$/, "");

export default function App() {
  return (
    <BrowserRouter basename={routerBasename}>
      <SessionForecastViewProvider>
        <AppRoutes />
      </SessionForecastViewProvider>
    </BrowserRouter>
  );
}
